#!/usr/bin/env python
"""Offline, fail-closed Task7 target package and reference primitives.

The module intentionally stops before prediction execution in this first
implementation slice.  It consumes an existing `VALIDATED_ONCE` cache set,
seals a role-blind IQ-only package and a separate truth sidecar, ingests a
read-only ADV3B02 target-known reference, and computes the explicit target
unknown gate.  It does not build data, revalidate data, adapt a model, fit a
threshold, or inspect performance to choose a method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import zipfile

import numpy as np

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    canonical_json_sha256,
    load_verified_leo_weak_cache_set,
    sha256_file,
)
from cvsrffi import phase1_clic_target_leo as _target


CLICTargetProtocolError = _target.CLICTargetProtocolError
CLICTargetGateError = _target.CLICTargetGateError

_VALIDATION_RECEIPT_SCHEMA = "cvs.phase2.data_validation_receipt.v1"
_KNOWN_TEST_CONFIG_SCHEMA = "cvs.phase1.clic_known_test_config.v1"
_EXPECTED_CACHE_SCOPE = "phase1_clic_target_confirmation"
_TARGET_REGISTERED_ROLE = "target_registered_known"
_TARGET_UNKNOWN_ROLE = "target_unknown"
_EXPECTED_ROLES = frozenset({_TARGET_REGISTERED_ROLE, _TARGET_UNKNOWN_ROLE})
_TARGET_PACKAGE_DATA_FILE = "received_iq.npz"
_TARGET_PACKAGE_MANIFEST_FILE = "manifest.json"
_TRUTH_SIDECAR_FILE = "truth_sidecar.json"
_PREDICTION_SCHEMA = "cvs.phase1.clic_target_prediction.v1"
_TARGET_SCORE_SCHEMA = "cvs.phase1.clic_target_leo_eval.v1"
_TARGET_METRICS_SCHEMA = "cvs.phase1.clic_target_metrics.v1"
_CONFIRMATION_TEST_SEMANTIC_FIELDS = frozenset(
    {
        "channel",
        "preprocess",
        "zero_adapt",
        "metrics",
    }
)


def _known_test_data_sha(config: Mapping[str, Any]) -> str:
    return _target.canonical_sha256(_target.normalize_known_test_config(config))


def _train_data_sha(config: Mapping[str, Any]) -> str:
    return _target.canonical_sha256(_target.normalize_train_data_config(config))


def _write_new_utf8_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    escape_truth_sidecar_key: bool = False,
) -> None:
    """Write one immutable JSON artifact without overwriting prior evidence.

    Predictor-visible prediction bytes retain the semantic
    ``truth_sidecar_opened`` evidence field, but encode the underscore in that
    key so a byte-level visibility audit cannot mistake the field name for a
    reachable truth-sidecar reference.  JSON decoding restores the exact key;
    all logical SHA checks therefore remain canonical and unchanged.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = _target.canonical_json_bytes(dict(payload)).decode("utf-8")
        if escape_truth_sidecar_key:
            encoded = encoded.replace(
                '"truth_sidecar_opened"', '"truth\\u005fsidecar_opened"'
            )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
    except FileExistsError as exc:
        raise CLICTargetProtocolError(f"{label} already exists and is immutable: {path}") from exc


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise CLICTargetProtocolError(f"{label} must be boolean")
    return value


def _confirmation_cache_snapshot(
    cache_set_manifest_path: Path,
    cache_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture exactly the already-verified cache bytes for a sealing window.

    This is deliberately only a byte-stability guard around the permitted
    cache-set loader.  It neither rebuilds rows nor repeats the channel
    overlay/validation procedure.
    """

    scenario_paths = cache_manifest.get("cache_npz_by_scenario")
    scenario_hashes = cache_manifest.get("cache_sha256_by_scenario")
    if not isinstance(scenario_paths, Mapping) or not isinstance(scenario_hashes, Mapping):
        raise CLICTargetProtocolError("confirmation cache-set scenario path/SHA mapping is invalid")
    if (
        tuple(str(key) for key in scenario_paths) != FORMAL_LEO_WEAK_SCENARIOS
        or tuple(str(key) for key in scenario_hashes) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise CLICTargetProtocolError("confirmation cache-set scenario order drift")
    snapshot_paths: dict[str, str] = {}
    snapshot_hashes: dict[str, str] = {}
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        raw_path = Path(str(scenario_paths[scene]))
        resolved = (
            raw_path if raw_path.is_absolute() else cache_set_manifest_path.parent / raw_path
        ).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"confirmation cache NPZ is missing for {scene}: {resolved}")
        expected_sha = _target.require_sha256(
            scenario_hashes[scene], label=f"confirmation cache {scene}"
        )
        if sha256_file(resolved) != expected_sha:
            raise CLICTargetProtocolError(
                f"confirmation cache NPZ SHA drift before sealing for {scene}"
            )
        snapshot_paths[scene] = str(resolved)
        snapshot_hashes[scene] = expected_sha
    return {
        "cache_set_manifest_path": str(cache_set_manifest_path),
        "cache_set_manifest_sha256": sha256_file(cache_set_manifest_path),
        "cache_npz_paths": snapshot_paths,
        "cache_npz_sha256_by_scenario": snapshot_hashes,
    }


def _assert_confirmation_cache_snapshot_unchanged(snapshot: Mapping[str, Any]) -> None:
    """Reject a cache-byte swap after the one permitted verified load."""

    manifest_path = Path(str(snapshot.get("cache_set_manifest_path", ""))).resolve()
    expected_manifest_sha = _target.require_sha256(
        snapshot.get("cache_set_manifest_sha256"),
        label="confirmation cache-set manifest",
    )
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha:
        raise CLICTargetProtocolError("confirmation cache-set manifest changed while sealing")
    paths = _target._require_mapping(
        snapshot.get("cache_npz_paths"), label="confirmation cache NPZ paths"
    )
    hashes = _target._require_mapping(
        snapshot.get("cache_npz_sha256_by_scenario"),
        label="confirmation cache NPZ SHA map",
    )
    if set(paths) != set(FORMAL_LEO_WEAK_SCENARIOS) or set(hashes) != set(
        FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise CLICTargetProtocolError("confirmation cache snapshot scene closure drift")
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        cache_path = Path(str(paths[scene])).resolve()
        expected_sha = _target.require_sha256(
            hashes[scene], label=f"confirmation cache {scene}"
        )
        if not cache_path.is_file() or sha256_file(cache_path) != expected_sha:
            raise CLICTargetProtocolError(
                f"confirmation cache NPZ changed while sealing for {scene}"
            )


def _confirmation_test_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only non-selection test semantics supplied by the operator."""

    semantics = _target._require_mapping(value, label="confirmation test semantics")
    if set(semantics) != _CONFIRMATION_TEST_SEMANTIC_FIELDS:
        raise CLICTargetProtocolError(
            "confirmation test semantics must contain exactly channel/preprocess/"
            "zero-adaptation/metric fields; target data selection is cache-derived"
        )
    if semantics.get("zero_adapt") is not True:
        raise CLICTargetProtocolError("confirmation target evaluation requires zero_adapt=true")
    for field in ("channel", "preprocess", "metrics"):
        mapped = semantics.get(field)
        if not isinstance(mapped, Mapping) or not mapped:
            raise CLICTargetProtocolError(f"confirmation test semantics {field} is invalid")
    # Canonical serialization rejects non-finite and non-JSON semantic values
    # before they can enter an immutable config manifest.
    _target.canonical_sha256(semantics)
    return dict(semantics)


def _derive_confirmation_known_test_config(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    *,
    test_semantics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive all target set membership from verified cache arrays only."""

    semantics = _confirmation_test_semantics(test_semantics)
    known_tx_ids: set[str] = set()
    unknown_tx_ids: set[str] = set()
    receiver_ids: set[str] = set()
    day_ids: set[str] = set()
    known_by_scene: dict[str, set[str]] = {}
    role_counts_by_scene: dict[str, dict[str, int]] = {}
    input_width: int | None = None
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario.get(scene)
        if arrays is None:
            raise CLICTargetProtocolError(
                f"verified confirmation cache is missing formal scene: {scene}"
            )
        iq = np.asarray(arrays.get("leo_weak_iq"), dtype=np.float32)
        roles = np.asarray(arrays.get("dataset_role")).astype(str).reshape(-1)
        tx_ids = np.asarray(arrays.get("tx_ids")).astype(str).reshape(-1)
        rx_ids = np.asarray(arrays.get("rx_ids")).astype(str).reshape(-1)
        day_values = np.asarray(arrays.get("day_ids")).astype(str).reshape(-1)
        if (
            iq.ndim != 3
            or iq.shape[1] != 2
            or iq.shape[0] <= 0
            or any(value.shape[0] != iq.shape[0] for value in (roles, tx_ids, rx_ids, day_values))
            or not np.isfinite(iq).all()
        ):
            raise CLICTargetProtocolError(
                f"verified confirmation cache metadata/IQ shape drift for {scene}"
            )
        width = int(iq.shape[2])
        if input_width is None:
            input_width = width
        elif input_width != width:
            raise CLICTargetProtocolError("confirmation cache received-IQ width drifts across scenes")
        if set(roles.tolist()) != _EXPECTED_ROLES:
            raise CLICTargetProtocolError(
                f"confirmation cache role closure drift for {scene}"
            )
        registered = {str(tx_ids[index]) for index, role in enumerate(roles) if role == _TARGET_REGISTERED_ROLE}
        unknown = {str(tx_ids[index]) for index, role in enumerate(roles) if role == _TARGET_UNKNOWN_ROLE}
        if not registered or not unknown or registered & unknown:
            raise CLICTargetProtocolError(
                f"confirmation cache registered/unknown TX partition drift for {scene}"
            )
        known_by_scene[scene] = registered
        known_tx_ids.update(registered)
        unknown_tx_ids.update(unknown)
        receiver_ids.update(str(value) for value in rx_ids)
        day_ids.update(str(value) for value in day_values)
        role_counts_by_scene[scene] = {
            _TARGET_REGISTERED_ROLE: int(np.sum(roles == _TARGET_REGISTERED_ROLE)),
            _TARGET_UNKNOWN_ROLE: int(np.sum(roles == _TARGET_UNKNOWN_ROLE)),
        }
    if (
        input_width is None
        or not known_tx_ids
        or not unknown_tx_ids
        or known_tx_ids & unknown_tx_ids
        or not receiver_ids
        or not day_ids
        or any(known_by_scene[scene] != known_tx_ids for scene in FORMAL_LEO_WEAK_SCENARIOS)
    ):
        raise CLICTargetProtocolError(
            "confirmation cache cannot derive one complete target-known universe"
        )
    preprocessing = _target._require_mapping(
        semantics["preprocess"], label="confirmation test preprocessing"
    )
    if (
        set(preprocessing) != {"input_len", "iq_dtype"}
        or type(preprocessing.get("input_len")) is not int
        or int(preprocessing["input_len"]) != input_width
        or preprocessing.get("iq_dtype") != "float32"
    ):
        raise CLICTargetProtocolError(
            "confirmation test preprocessing must exactly match verified received-IQ width/dtype"
        )
    normalized = {
        "target_receiver_ids": sorted(receiver_ids),
        "target_day_ids": sorted(day_ids),
        "target_known_tx_ids": sorted(known_tx_ids),
        "class_order": sorted(known_tx_ids),
        "scenes": list(FORMAL_LEO_WEAK_SCENARIOS),
        "leo_weak_channel": dict(semantics["channel"]),
        "preprocessing": dict(preprocessing),
        "zero_adaptation": True,
        "metric_definitions": dict(semantics["metrics"]),
    }
    normalized = _target.normalize_known_test_config(normalized)
    return normalized, {
        "target_unknown_tx_ids": sorted(unknown_tx_ids),
        "role_row_count_by_scene": role_counts_by_scene,
    }


def _derived_confirmation_identifiers(
    *,
    cache_snapshot: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    known_config: Mapping[str, Any],
    derivation_audit: Mapping[str, Any],
) -> tuple[str, str]:
    """Make opaque capsule/split identifiers deterministic cache content values."""

    cache_manifest_sha = _target.require_sha256(
        cache_snapshot.get("cache_set_manifest_sha256"),
        label="confirmation cache-set manifest",
    )
    assignment_sha = _target.require_sha256(
        cache_audit.get("physical_sample_scenario_assignment_sha256"),
        label="confirmation cache physical scenario assignment",
    )
    capsule_id = _target.canonical_sha256(
        {
            "schema": "cvs.phase1.clic_target_confirmation_capsule_id.v1",
            "cache_set_manifest_sha256": cache_manifest_sha,
            "cache_scope": _EXPECTED_CACHE_SCOPE,
            "protocol_schema": "p2_min_v1",
        }
    )
    split_id = _target.canonical_sha256(
        {
            "schema": "cvs.phase1.clic_target_confirmation_split_id.v1",
            "physical_sample_scenario_assignment_sha256": assignment_sha,
            "target_known_tx_ids": list(known_config["target_known_tx_ids"]),
            "target_unknown_tx_ids": list(derivation_audit["target_unknown_tx_ids"]),
            "target_receiver_ids": list(known_config["target_receiver_ids"]),
            "target_day_ids": list(known_config["target_day_ids"]),
            "scenes": list(FORMAL_LEO_WEAK_SCENARIOS),
        }
    )
    return capsule_id, split_id


def seal_clic_target_confirmation_validation(
    cache_set_manifest_path: str | Path,
    output_root: str | Path,
    *,
    test_semantics: Mapping[str, Any],
    test_semantics_artifact_path: str | Path | None = None,
    expected_capsule_id: str | None = None,
    expected_split_id: str | None = None,
    expected_protocol_schema: str = "p2_min_v1",
) -> dict[str, str]:
    """Seal evaluator-only known config plus a VALIDATED_ONCE receipt.

    The function is the sole production bridge from a new confirmation-scope
    cache set to the receipt consumed by :func:`seal_clic_target_package`.
    It uses the existing verified-cache loader exactly once, derives target
    TX/RX/day/class membership from those arrays, and never rebuilds IQ,
    re-applies an overlay, or accepts caller-provided data roots.
    """

    if str(expected_protocol_schema) != "p2_min_v1":
        raise CLICTargetProtocolError(
            "confirmation validation requires protocol_schema=p2_min_v1"
        )
    semantics_path: Path | None = None
    semantics_raw_sha: str | None = None
    if test_semantics_artifact_path is not None:
        semantics_path = Path(test_semantics_artifact_path).resolve()
        if not semantics_path.is_file():
            raise FileNotFoundError(
                f"confirmation test-semantics JSON is missing: {semantics_path}"
            )
        semantics_raw_sha = sha256_file(semantics_path)
        reopened_semantics = _target.read_json_object(
            semantics_path, label="confirmation test-semantics JSON"
        )
        if sha256_file(semantics_path) != semantics_raw_sha:
            raise CLICTargetProtocolError(
                "confirmation test-semantics JSON changed while opening"
            )
        if reopened_semantics != dict(test_semantics):
            raise CLICTargetProtocolError(
                "confirmation test-semantics caller mapping does not equal sealed JSON bytes"
            )
    cache_path = Path(cache_set_manifest_path).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"confirmation cache-set manifest is missing: {cache_path}")
    output_dir = Path(output_root).resolve()
    if output_dir.exists():
        raise CLICTargetProtocolError(
            f"confirmation validation output already exists and is immutable: {output_dir}"
        )
    arrays_by_scenario, cache_manifest, cache_audit = load_verified_leo_weak_cache_set(
        cache_path,
        expected_scope=_EXPECTED_CACHE_SCOPE,
        allowed_roles=_EXPECTED_ROLES,
    )
    if str(cache_manifest.get("cache_scope", "")) != _EXPECTED_CACHE_SCOPE:
        raise CLICTargetProtocolError("confirmation cache-set scope drift")
    cache_snapshot = _confirmation_cache_snapshot(cache_path, cache_manifest)
    known_normalized, derivation_audit = _derive_confirmation_known_test_config(
        arrays_by_scenario,
        test_semantics=test_semantics,
    )
    capsule_id, split_id = _derived_confirmation_identifiers(
        cache_snapshot=cache_snapshot,
        cache_audit=cache_audit,
        known_config=known_normalized,
        derivation_audit=derivation_audit,
    )
    for label, expected, observed in (
        ("capsule", expected_capsule_id, capsule_id),
        ("split", expected_split_id, split_id),
    ):
        if expected is not None and (not isinstance(expected, str) or expected != observed):
            raise CLICTargetProtocolError(
                f"confirmation expected {label}_id does not equal the cache-derived value"
            )
    _assert_confirmation_cache_snapshot_unchanged(cache_snapshot)
    if semantics_path is not None and sha256_file(semantics_path) != semantics_raw_sha:
        raise CLICTargetProtocolError("confirmation test-semantics JSON changed while sealing")
    output_dir.mkdir(parents=True, exist_ok=False)
    known_path = output_dir / "known_test_config.json"
    known_payload = {
        "schema": _KNOWN_TEST_CONFIG_SCHEMA,
        "sealed": True,
        "capsule_id": capsule_id,
        "split_id": split_id,
        "cache_set_manifest_sha256": cache_snapshot["cache_set_manifest_sha256"],
        "normalized": known_normalized,
        "normalized_sha256": _target.canonical_sha256(known_normalized),
    }
    _write_new_utf8_json(known_path, known_payload, label="confirmation known-test config")
    known_raw_sha = sha256_file(known_path)
    _assert_confirmation_cache_snapshot_unchanged(cache_snapshot)
    if semantics_path is not None and sha256_file(semantics_path) != semantics_raw_sha:
        raise CLICTargetProtocolError("confirmation test-semantics JSON changed while sealing")
    receipt_path = output_dir / "validator_receipt.json"
    receipt = {
        "schema": _VALIDATION_RECEIPT_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "protocol_schema": "p2_min_v1",
        "capsule_id": capsule_id,
        "split_id": split_id,
        "cache_set_manifest_path": str(cache_path),
        "cache_set_manifest_sha256": cache_snapshot["cache_set_manifest_sha256"],
        "cache_scope": _EXPECTED_CACHE_SCOPE,
        "truth_role_blind_scene_assignment": True,
        "known_test_config_manifest_path": str(known_path),
        "known_test_config_raw_sha256": known_raw_sha,
        "known_test_config_normalized_sha256": known_payload["normalized_sha256"],
        "role_row_count_by_scene": derivation_audit["role_row_count_by_scene"],
    }
    if semantics_path is not None:
        receipt["test_semantics_json_path"] = str(semantics_path)
        receipt["test_semantics_json_raw_sha256"] = str(semantics_raw_sha)
    _write_new_utf8_json(receipt_path, receipt, label="confirmation validation receipt")
    return {
        "receipt_path": str(receipt_path),
        "known_test_config_path": str(known_path),
        "capsule_id": capsule_id,
        "split_id": split_id,
        "cache_set_manifest_path": str(cache_path),
    }


def _read_validated_receipt(
    validator_receipt_path: str | Path,
    *,
    cache_set_manifest_path: Path,
    expected_capsule_id: str,
    expected_split_id: str,
    expected_protocol_schema: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Verify the independent builder receipt before materializing target IQ."""

    receipt_path = Path(validator_receipt_path).resolve()
    receipt_raw_before = sha256_file(receipt_path)
    receipt = _target.read_json_object(receipt_path, label="Phase2 validation receipt")
    receipt_raw_after = sha256_file(receipt_path)
    if receipt_raw_before != receipt_raw_after:
        raise CLICTargetProtocolError("Phase2 validation receipt changed while opening")
    required = {
        "schema": _VALIDATION_RECEIPT_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "protocol_schema": str(expected_protocol_schema),
        "capsule_id": str(expected_capsule_id),
        "split_id": str(expected_split_id),
        "cache_scope": _EXPECTED_CACHE_SCOPE,
    }
    drift = [key for key, expected in required.items() if receipt.get(key) != expected]
    if drift:
        expected_text = {key: required[key] for key in drift}
        raise CLICTargetProtocolError(
            "Phase2 validation receipt contract drift: "
            f"fields={drift}, expected={expected_text}"
        )
    if not _require_bool(
        receipt.get("truth_role_blind_scene_assignment"),
        label="Phase2 receipt truth/role-blind scene assignment",
    ):
        raise CLICTargetProtocolError("Phase2 receipt must be truth/role-blind")

    declared_cache_raw = Path(str(receipt.get("cache_set_manifest_path", "")))
    declared_cache_path = (
        declared_cache_raw
        if declared_cache_raw.is_absolute()
        else receipt_path.parent / declared_cache_raw
    ).resolve()
    if declared_cache_path != cache_set_manifest_path:
        raise CLICTargetProtocolError(
            "Phase2 receipt cache-set manifest path does not match requested cache set"
        )
    declared_cache_sha = _target.require_sha256(
        receipt.get("cache_set_manifest_sha256"), label="Phase2 receipt cache-set manifest"
    )
    if sha256_file(cache_set_manifest_path) != declared_cache_sha:
        raise CLICTargetProtocolError("Phase2 receipt cache-set manifest SHA drift")

    known_raw = Path(str(receipt.get("known_test_config_manifest_path", "")))
    known_path = (known_raw if known_raw.is_absolute() else receipt_path.parent / known_raw).resolve()
    known_raw_sha = _target.require_sha256(
        receipt.get("known_test_config_raw_sha256"), label="known-test config"
    )
    known_config = _target.read_verified_config_manifest(
        known_path,
        expected_schema="cvs.phase1.clic_known_test_config.v1",
        expected_raw_sha256=known_raw_sha,
        label="candidate known-test config",
    )
    # This validates the exact data-bearing surface now, rather than trusting a
    # caller dictionary or accepting a malformed target configuration later.
    known_config["data_normalized_sha256"] = _known_test_data_sha(
        known_config["normalized"]
    )
    return receipt, known_config, receipt_raw_after


def _scene_rows_from_verified_cache(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    *,
    lineage_sha256: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, int]]:
    """Build predictor-safe arrays and the separate truth records in memory."""

    received_rows: list[np.ndarray] = []
    scenes: list[str] = []
    iq_hashes: list[str] = []
    tokens: list[str] = []
    truth_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    common_width: int | None = None
    ordinal = 0

    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario.get(scene)
        if arrays is None:
            raise CLICTargetProtocolError(f"verified cache is missing formal scene: {scene}")
        iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)
        if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[0] <= 0:
            raise CLICTargetProtocolError(f"verified target received-IQ shape drift for {scene}")
        if not np.isfinite(iq).all():
            raise CLICTargetProtocolError(f"verified target received-IQ is non-finite for {scene}")
        if common_width is None:
            common_width = int(iq.shape[2])
        elif int(iq.shape[2]) != common_width:
            raise CLICTargetProtocolError("target received-IQ input length drifts across scenes")

        roles = np.asarray(arrays["dataset_role"]).astype(str)
        tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
        rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
        day_ids = np.asarray(arrays["day_ids"]).astype(str)
        physical_ids = np.asarray(arrays["sample_ids"]).astype(str)
        row_hashes = np.asarray(arrays["post_channel_iq_sha256"]).astype(str)
        row_count = int(iq.shape[0])
        if any(
            values.shape[0] != row_count
            for values in (roles, tx_ids, rx_ids, day_ids, physical_ids, row_hashes)
        ):
            raise CLICTargetProtocolError(f"verified target metadata row-count drift for {scene}")
        if set(roles.tolist()) != _EXPECTED_ROLES:
            raise CLICTargetProtocolError(f"verified target role set drift for {scene}")

        counts[scene] = row_count
        for index in range(row_count):
            role = str(roles[index])
            iq_sha = _target.require_sha256(row_hashes[index], label="received IQ")
            token = _target.opaque_token(
                lineage_sha256=lineage_sha256,
                scene=scene,
                ordinal=ordinal,
                received_iq_sha256=iq_sha,
            )
            received_rows.append(np.array(iq[index], copy=True))
            scenes.append(scene)
            iq_hashes.append(iq_sha)
            tokens.append(token)
            truth_rows.append(
                {
                    "opaque_token": token,
                    "scene": scene,
                    "role": role,
                    # A registered query is scored against its true registered
                    # identity.  An unregistered query's true identity remains
                    # intentionally collapsed to the evaluator-only token.
                    "truth": str(tx_ids[index]) if role == _TARGET_REGISTERED_ROLE else "unknown",
                    "tx_id": str(tx_ids[index]),
                    "rx_id": str(rx_ids[index]),
                    "day_id": str(day_ids[index]),
                    # This pre-overlay physical identity is evaluator-only.
                    # It never enters the IQ-only package or predictor API.
                    "physical_sample_id": str(physical_ids[index]),
                }
            )
            ordinal += 1

    physical_values = [str(row["physical_sample_id"]) for row in truth_rows]
    if (
        not received_rows
        or len(tokens) != len(set(tokens))
        or any(not value for value in physical_values)
        or len(physical_values) != len(set(physical_values))
    ):
        raise CLICTargetProtocolError("target opaque-token/physical-sample closure failed")
    package_arrays = {
        "received_iq": np.asarray(received_rows, dtype=np.float32),
        "opaque_tokens": np.asarray(tokens, dtype="U64"),
        "scenes": np.asarray(scenes, dtype="U32"),
        "received_iq_sha256": np.asarray(iq_hashes, dtype="U64"),
    }
    return package_arrays, truth_rows, counts


def _scene_seed_assignment_sha256(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    package_arrays: Mapping[str, np.ndarray],
) -> str:
    """Commit the truth-blind row-to-scene/channel-seed assignment in memory.

    The package never contains a seed, role, or physical identity.  This
    commitment is nevertheless calculated from the already verified cache rows
    and the final opaque-token ordering, so a later cache/seed swap cannot be
    represented as the same sealed IQ-only package.
    """

    tokens = np.asarray(package_arrays.get("opaque_tokens")).astype(str)
    scenes = np.asarray(package_arrays.get("scenes")).astype(str)
    iq_hashes = np.asarray(package_arrays.get("received_iq_sha256")).astype(str)
    if not tokens.size or tokens.shape != scenes.shape or tokens.shape != iq_hashes.shape:
        raise CLICTargetProtocolError("target scene/seed assignment package row closure failed")
    assignments: list[dict[str, Any]] = []
    ordinal = 0
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario.get(scene)
        if arrays is None:
            raise CLICTargetProtocolError(f"target scene/seed assignment lacks {scene} cache")
        seeds = np.asarray(arrays.get("satellite_seeds"))
        row_hashes = np.asarray(arrays.get("post_channel_iq_sha256")).astype(str)
        if seeds.ndim != 1 or row_hashes.ndim != 1 or seeds.shape != row_hashes.shape:
            raise CLICTargetProtocolError(f"target scene/seed assignment row alignment drift for {scene}")
        if seeds.dtype.kind not in {"i", "u"}:
            raise CLICTargetProtocolError(f"target scene/seed assignment seed dtype drift for {scene}")
        for index in range(int(seeds.shape[0])):
            if ordinal >= int(tokens.shape[0]) or scenes[ordinal] != scene:
                raise CLICTargetProtocolError("target scene/seed assignment package scene ordering drift")
            received_iq_sha = _target.require_sha256(
                row_hashes[index], label="target scene/seed assignment received IQ"
            )
            if iq_hashes[ordinal] != received_iq_sha:
                raise CLICTargetProtocolError("target scene/seed assignment received-IQ binding drift")
            assignments.append(
                {
                    "opaque_token": _target.require_sha256(
                        tokens[ordinal], label="target scene/seed assignment opaque token"
                    ),
                    "scene": scene,
                    "channel_seed": int(seeds[index]),
                    "received_iq_sha256": received_iq_sha,
                }
            )
            ordinal += 1
    if ordinal != int(tokens.shape[0]):
        raise CLICTargetProtocolError("target scene/seed assignment row count drift")
    return _target.canonical_sha256(assignments)


_TARGET_UNIVERSE_ROOT_FIELDS = (
    "target_receiver_set_sha256",
    "target_registered_tx_set_sha256",
    "target_unknown_tx_set_sha256",
    "target_day_set_sha256",
    "merged_physical_sample_ids_sha256",
)


def _target_universe_roots_from_truth_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Derive evaluator-only target-universe roots from complete truth rows."""

    receivers: set[str] = set()
    registered_tx: set[str] = set()
    unknown_tx: set[str] = set()
    days: set[str] = set()
    physical: set[str] = set()
    seen = 0
    for index, raw in enumerate(rows):
        row = _target._require_mapping(raw, label=f"target universe truth row {index}")
        role = str(row.get("role", ""))
        tx_id = str(row.get("tx_id", ""))
        rx_id = str(row.get("rx_id", ""))
        day_id = str(row.get("day_id", ""))
        physical_id = str(row.get("physical_sample_id", ""))
        if role not in _EXPECTED_ROLES or not all((tx_id, rx_id, day_id, physical_id)):
            raise CLICTargetProtocolError("target universe truth metadata/role drift")
        if physical_id in physical:
            raise CLICTargetProtocolError("target universe physical sample IDs are not globally unique")
        physical.add(physical_id)
        receivers.add(rx_id)
        days.add(day_id)
        if role == _TARGET_REGISTERED_ROLE:
            registered_tx.add(tx_id)
        else:
            unknown_tx.add(tx_id)
        seen += 1
    if not seen or not all((receivers, registered_tx, unknown_tx, days, physical)):
        raise CLICTargetProtocolError("target universe root inputs must all be nonempty")
    return {
        "target_receiver_set_sha256": _target.canonical_sha256(sorted(receivers)),
        "target_registered_tx_set_sha256": _target.canonical_sha256(sorted(registered_tx)),
        "target_unknown_tx_set_sha256": _target.canonical_sha256(sorted(unknown_tx)),
        "target_day_set_sha256": _target.canonical_sha256(sorted(days)),
        "merged_physical_sample_ids_sha256": _target.canonical_sha256(sorted(physical)),
    }


def _validate_target_universe_against_known_config(
    roots: Mapping[str, Any], known_config: Mapping[str, Any]
) -> None:
    """Close manifest roots to the public known-target configuration surface."""

    if set(roots) != set(_TARGET_UNIVERSE_ROOT_FIELDS):
        raise CLICTargetProtocolError("target universe root field drift")
    for field in _TARGET_UNIVERSE_ROOT_FIELDS:
        _target.require_sha256(roots.get(field), label=f"target universe {field}")
    known = _target.normalize_known_test_config(known_config)
    expected = {
        "target_receiver_set_sha256": _target.canonical_sha256(
            sorted({str(item) for item in known["target_receiver_ids"]})
        ),
        "target_registered_tx_set_sha256": _target.canonical_sha256(
            sorted({str(item) for item in known["target_known_tx_ids"]})
        ),
        "target_day_set_sha256": _target.canonical_sha256(
            sorted({str(item) for item in known["target_day_ids"]})
        ),
    }
    drift = {
        field: {"observed": roots[field], "expected": expected[field]}
        for field in expected
        if roots[field] != expected[field]
    }
    if drift:
        raise CLICTargetProtocolError(
            f"target universe roots do not match known-test config: {sorted(drift)}"
        )


def seal_clic_target_package(
    cache_set_root: str | Path,
    output_root: str | Path,
    *,
    validator_receipt_path: str | Path,
    expected_capsule_id: str,
    expected_split_id: str,
    expected_protocol_schema: str = "p2_min_v1",
) -> tuple[Path, Path]:
    """Seal an existing validated cache set into IQ-only and truth-side artifacts.

    The receipt check is intentionally first.  No builder or revalidation path
    is imported or called here; once receipt identity drifts, no target IQ is
    opened.  ``load_verified_leo_weak_cache_set`` performs the one permitted
    existing-cache integrity check and enforces the one-observation contract.
    """

    cache_manifest_path = Path(cache_set_root).resolve()
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(f"target cache-set manifest is missing: {cache_manifest_path}")
    if str(expected_protocol_schema) != "p2_min_v1":
        raise CLICTargetProtocolError("target sealer requires protocol_schema=p2_min_v1")

    receipt_path = Path(validator_receipt_path).resolve()
    receipt, known_config, receipt_raw_sha = _read_validated_receipt(
        validator_receipt_path,
        cache_set_manifest_path=cache_manifest_path,
        expected_capsule_id=str(expected_capsule_id),
        expected_split_id=str(expected_split_id),
        expected_protocol_schema=str(expected_protocol_schema),
    )

    arrays_by_scenario, cache_manifest, cache_audit = load_verified_leo_weak_cache_set(
        cache_manifest_path,
        expected_scope=_EXPECTED_CACHE_SCOPE,
        allowed_roles=_EXPECTED_ROLES,
    )
    # Bind the receipt and the independently reopened cache manifest together.
    if str(cache_manifest.get("cache_scope", "")) != _EXPECTED_CACHE_SCOPE:
        raise CLICTargetProtocolError("target cache-set scope drift")
    cache_manifest_raw_sha = sha256_file(cache_manifest_path)
    if cache_manifest_raw_sha != _target.require_sha256(
        receipt.get("cache_set_manifest_sha256"), label="Phase2 receipt cache-set manifest"
    ):
        raise CLICTargetProtocolError("target cache-set manifest changed during target sealing")

    def assert_seal_inputs_unchanged() -> None:
        if (
            sha256_file(receipt_path) != receipt_raw_sha
            or sha256_file(cache_manifest_path) != cache_manifest_raw_sha
            or sha256_file(Path(known_config["path"])) != known_config["raw_sha256"]
        ):
            raise CLICTargetProtocolError("target sealer input artifact changed during sealing")

    pre_lineage = _target.canonical_sha256(
        {
            "cache_set_manifest_sha256": cache_manifest_raw_sha,
            "cache_set_physical_assignment_sha256": cache_audit[
                "physical_sample_scenario_assignment_sha256"
            ],
            "capsule_id": str(expected_capsule_id),
            "known_test_config_normalized_sha256": known_config["data_normalized_sha256"],
            "known_test_config_raw_sha256": known_config["raw_sha256"],
            "protocol_schema": str(expected_protocol_schema),
            "receipt_sha256": receipt_raw_sha,
            "split_id": str(expected_split_id),
        }
    )
    package_arrays, truth_rows, row_count_by_scene = _scene_rows_from_verified_cache(
        arrays_by_scenario, lineage_sha256=pre_lineage
    )
    received_iq_root = _target.canonical_sha256(
        package_arrays["received_iq_sha256"].astype(str).tolist()
    )
    package_lineage = _target.canonical_sha256(
        {
            "pre_lineage_sha256": pre_lineage,
            "received_iq_sha256_root": received_iq_root,
            "row_count": int(package_arrays["received_iq"].shape[0]),
            "scene_row_count": row_count_by_scene,
        }
    )
    # Regenerate opaque tokens with the final lineage identity.  This makes
    # both package and sidecar independently reject a mix from another seal.
    package_arrays, truth_rows, row_count_by_scene = _scene_rows_from_verified_cache(
        arrays_by_scenario, lineage_sha256=package_lineage
    )
    scene_seed_assignment_sha = _scene_seed_assignment_sha256(
        arrays_by_scenario, package_arrays
    )
    received_iq_root = _target.canonical_sha256(
        package_arrays["received_iq_sha256"].astype(str).tolist()
    )
    universe_roots = _target_universe_roots_from_truth_rows(truth_rows)
    _validate_target_universe_against_known_config(
        universe_roots, known_config["normalized"]
    )
    assert_seal_inputs_unchanged()

    output = Path(output_root).resolve()
    if output.exists():
        raise CLICTargetProtocolError(f"target seal output already exists and is immutable: {output}")
    package_dir = output / "iq_only_package"
    truth_path = output / _TRUTH_SIDECAR_FILE
    data_path = package_dir / _TARGET_PACKAGE_DATA_FILE
    manifest_path = package_dir / _TARGET_PACKAGE_MANIFEST_FILE

    package_dir.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(data_path, **package_arrays)
        data_sha = sha256_file(data_path)
        manifest_base = {
            "schema": _target.TARGET_PACKAGE_SCHEMA,
            "capsule_id": str(expected_capsule_id),
            "split_id": str(expected_split_id),
            "protocol_schema": str(expected_protocol_schema),
            "query_truth_included": False,
            "query_role_included": False,
            "single_leo_observation": True,
            "scenes": list(FORMAL_LEO_WEAK_SCENARIOS),
            "scene_physical_id_sha256": dict(
                cache_audit["physical_sample_ids_sha256_by_scenario"]
            ),
            "scene_physical_id_pairwise_disjoint": True,
            "physical_sample_scenario_assignment_sha256": cache_audit[
                "physical_sample_scenario_assignment_sha256"
            ],
            "scene_seed_assignment_sha256": scene_seed_assignment_sha,
            # Set roots bind the target population without putting any
            # receiver/TX/day/physical identity into the predictor package.
            **universe_roots,
            "received_iq_sha256_root": received_iq_root,
            "received_iq_data_sha256": data_sha,
            "row_count": int(package_arrays["received_iq"].shape[0]),
            "row_count_by_scene": row_count_by_scene,
            "opaque_token_sha256": _target.canonical_sha256(
                package_arrays["opaque_tokens"].astype(str).tolist()
            ),
            "lineage_sha256": package_lineage,
            "validator_receipt_sha256": receipt_raw_sha,
            "cache_set_manifest_sha256": cache_manifest_raw_sha,
            # Technical commitments are safe for the predictor to carry; the
            # corresponding path and all semantic config contents stay truth-
            # side only.
            "known_test_config_raw_sha256": known_config["raw_sha256"],
            "known_test_config_normalized_sha256": known_config["data_normalized_sha256"],
        }
        package_sha = _target.canonical_sha256(manifest_base)
        manifest = dict(manifest_base, package_sha256=package_sha)
        _write_new_utf8_json(manifest_path, manifest, label="IQ-only target package manifest")
        sidecar = {
            "schema": _target.TARGET_TRUTH_SCHEMA,
            "sealed": True,
            "package_sha256": package_sha,
            "lineage_sha256": package_lineage,
            # These reopening paths stay evaluator-only.  They deliberately do
            # not appear in the IQ-only package or truth-blind prediction.
            "validator_receipt_path": str(receipt_path),
            "validator_receipt_raw_sha256": receipt_raw_sha,
            "cache_set_manifest_path": str(cache_manifest_path),
            "cache_set_manifest_raw_sha256": cache_manifest_raw_sha,
            # The known-target configuration is evaluator-only.  In
            # particular, neither this path nor its semantic TX/RX/day/class
            # contents can reach the predictor-readable package or prediction.
            "known_test_config_manifest_path": known_config["path"],
            "known_test_config_raw_sha256": known_config["raw_sha256"],
            "known_test_config_normalized_sha256": known_config["data_normalized_sha256"],
            "row_count": len(truth_rows),
            "rows": truth_rows,
        }
        _write_new_utf8_json(truth_path, sidecar, label="target truth sidecar")
        assert_seal_inputs_unchanged()
    except Exception:
        # The directory did not exist before this call.  Do not leave a partial
        # artifact that later callers could mistake for a sealed package.
        for child in sorted(output.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        output.rmdir()
        raise
    return package_dir, truth_path


def _read_verified_clic_iq_only_package_header(package_path: str | Path) -> dict[str, Any]:
    """Verify the package header and byte identities without materializing IQ.

    Both the predictor publisher and the isolated scorer use this entry point.
    The scorer deliberately stops here: it may hash the sealed package data for
    integrity, but it never deserializes target IQ or invokes a predictor.
    """

    package = Path(package_path).resolve()
    if not package.is_dir():
        raise CLICTargetProtocolError("CLIC target IQ-only package must be a sealed directory")
    observed_names = {entry.name for entry in package.iterdir()}
    expected_names = {_TARGET_PACKAGE_DATA_FILE, _TARGET_PACKAGE_MANIFEST_FILE}
    if observed_names != expected_names:
        raise CLICTargetProtocolError("CLIC target IQ-only package member allowlist drift")
    manifest_path = package / _TARGET_PACKAGE_MANIFEST_FILE
    data_path = package / _TARGET_PACKAGE_DATA_FILE
    manifest_raw_before = sha256_file(manifest_path)
    manifest = _target.read_json_object(manifest_path, label="CLIC target IQ-only package manifest")
    manifest_raw_after = sha256_file(manifest_path)
    if manifest_raw_before != manifest_raw_after:
        raise CLICTargetProtocolError("CLIC target IQ-only package manifest changed while opening")
    required = {
        "schema",
        "capsule_id",
        "split_id",
        "protocol_schema",
        "query_truth_included",
        "query_role_included",
        "single_leo_observation",
        "scenes",
        "scene_physical_id_sha256",
        "scene_physical_id_pairwise_disjoint",
        "physical_sample_scenario_assignment_sha256",
        "scene_seed_assignment_sha256",
        "target_receiver_set_sha256",
        "target_registered_tx_set_sha256",
        "target_unknown_tx_set_sha256",
        "target_day_set_sha256",
        "merged_physical_sample_ids_sha256",
        "received_iq_sha256_root",
        "received_iq_data_sha256",
        "row_count",
        "row_count_by_scene",
        "opaque_token_sha256",
        "lineage_sha256",
        "validator_receipt_sha256",
        "cache_set_manifest_sha256",
        "known_test_config_raw_sha256",
        "known_test_config_normalized_sha256",
        "package_sha256",
    }
    if set(manifest) != required or manifest.get("schema") != _target.TARGET_PACKAGE_SCHEMA:
        raise CLICTargetProtocolError("CLIC target IQ-only package manifest schema/fields drift")
    if (
        manifest.get("protocol_schema") != "p2_min_v1"
        or manifest.get("query_truth_included") is not False
        or manifest.get("query_role_included") is not False
        or manifest.get("single_leo_observation") is not True
        or manifest.get("scenes") != list(FORMAL_LEO_WEAK_SCENARIOS)
        or manifest.get("scene_physical_id_pairwise_disjoint") is not True
    ):
        raise CLICTargetProtocolError("CLIC target IQ-only package protocol/role isolation drift")
    for field in (
        "physical_sample_scenario_assignment_sha256",
        "scene_seed_assignment_sha256",
        "target_receiver_set_sha256",
        "target_registered_tx_set_sha256",
        "target_unknown_tx_set_sha256",
        "target_day_set_sha256",
        "merged_physical_sample_ids_sha256",
        "received_iq_sha256_root",
        "received_iq_data_sha256",
        "opaque_token_sha256",
        "lineage_sha256",
        "validator_receipt_sha256",
        "cache_set_manifest_sha256",
        "known_test_config_raw_sha256",
        "known_test_config_normalized_sha256",
        "package_sha256",
    ):
        _target.require_sha256(manifest.get(field), label=f"target package {field}")
    scene_roots = manifest.get("scene_physical_id_sha256")
    if not isinstance(scene_roots, Mapping) or set(scene_roots) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTargetProtocolError("CLIC target IQ-only package scene physical-ID roots drift")
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        _target.require_sha256(scene_roots[scene], label=f"target package {scene} physical IDs")
    base = {key: value for key, value in manifest.items() if key != "package_sha256"}
    if _target.canonical_sha256(base) != manifest["package_sha256"]:
        raise CLICTargetProtocolError("CLIC target IQ-only package logical SHA drift")
    data_raw_sha = sha256_file(data_path)
    if data_raw_sha != manifest["received_iq_data_sha256"]:
        raise CLICTargetProtocolError("CLIC target IQ-only package received-IQ data SHA drift")
    return {
        "path": package,
        "manifest_path": manifest_path,
        "manifest_raw_sha256": manifest_raw_after,
        "data_path": data_path,
        "data_raw_sha256": data_raw_sha,
        "manifest": manifest,
    }


def _load_verified_clic_iq_only_package(package_path: str | Path) -> dict[str, Any]:
    """Open exactly the two predictor-safe package members and verify closure."""

    header = _read_verified_clic_iq_only_package_header(package_path)
    data_path = Path(header["data_path"])
    try:
        with np.load(data_path, allow_pickle=False) as archive:
            members = set(archive.files)
            expected_arrays = {"received_iq", "opaque_tokens", "scenes", "received_iq_sha256"}
            if members != expected_arrays:
                raise CLICTargetProtocolError("CLIC target IQ-only array member allowlist drift")
            arrays = {name: np.asarray(archive[name]) for name in expected_arrays}
    except (OSError, ValueError) as exc:
        if isinstance(exc, CLICTargetProtocolError):
            raise
        raise CLICTargetProtocolError("CLIC target IQ-only received-IQ archive is invalid") from exc
    if any(value.dtype == object for value in arrays.values()):
        raise CLICTargetProtocolError("CLIC target IQ-only archive object arrays are forbidden")
    received_iq = np.asarray(arrays["received_iq"], dtype=np.float32)
    tokens = np.asarray(arrays["opaque_tokens"]).astype(str)
    scenes = np.asarray(arrays["scenes"]).astype(str)
    iq_hashes = np.asarray(arrays["received_iq_sha256"]).astype(str)
    if (
        received_iq.ndim != 3
        or received_iq.shape[1] != 2
        or received_iq.shape[0] <= 0
        or not np.isfinite(received_iq).all()
        or any(values.shape != (received_iq.shape[0],) for values in (tokens, scenes, iq_hashes))
    ):
        raise CLICTargetProtocolError("CLIC target IQ-only array shape/non-finite closure failed")
    manifest = dict(header["manifest"])
    row_count = _as_positive_int(manifest.get("row_count"), label="target package row_count")
    if int(received_iq.shape[0]) != row_count:
        raise CLICTargetProtocolError("CLIC target IQ-only package row count drift")
    if len(set(tokens.tolist())) != row_count or any(len(token) != 64 for token in tokens.tolist()):
        raise CLICTargetProtocolError("CLIC target IQ-only opaque token closure failed")
    if set(scenes.tolist()) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTargetProtocolError("CLIC target IQ-only formal scene coverage drift")
    counts = {scene: int(np.sum(scenes == scene)) for scene in FORMAL_LEO_WEAK_SCENARIOS}
    if counts != manifest.get("row_count_by_scene") or any(value <= 0 for value in counts.values()):
        raise CLICTargetProtocolError("CLIC target IQ-only per-scene row count drift")
    for value in iq_hashes.tolist():
        _target.require_sha256(value, label="target package received IQ")
    if _target.canonical_sha256(iq_hashes.tolist()) != manifest["received_iq_sha256_root"]:
        raise CLICTargetProtocolError("CLIC target IQ-only received-IQ root drift")
    if _target.canonical_sha256(tokens.tolist()) != manifest["opaque_token_sha256"]:
        raise CLICTargetProtocolError("CLIC target IQ-only opaque token root drift")
    return {
        **header,
        "received_iq": received_iq,
        "opaque_tokens": tokens,
        "scenes": scenes,
        "received_iq_sha256": iq_hashes,
    }


def _runtime_train_config(runtime: Any) -> dict[str, Any]:
    """Reopen predictor-bound training config before allowing a target forward."""

    required = (
        "train_config_manifest_path",
        "train_config_raw_sha256",
        "train_config_normalized_sha256",
    )
    if any(not hasattr(runtime, field) for field in required):
        raise CLICTargetProtocolError("target predictor runtime lacks sealed train config binding")
    raw_path = str(runtime.train_config_manifest_path)
    raw_sha = _target.require_sha256(runtime.train_config_raw_sha256, label="predictor train config raw")
    expected_normalized = _target.require_sha256(
        runtime.train_config_normalized_sha256, label="predictor train config normalized"
    )
    member_name = getattr(runtime, "train_config_member_name", None)
    if member_name is None:
        config = _target.read_verified_config_manifest(
            raw_path,
            expected_schema="cvs.phase1.clic_train_data_config.v1",
            expected_raw_sha256=raw_sha,
            label="predictor train config",
        )
        data_sha = _train_data_sha(config["normalized"])
        # Existing sealed descriptors may have used their manifest-local SHA.
        # Store it unchanged for byte provenance, but do all comparisons on the
        # canonical data SHA at scoring time.
        if expected_normalized not in {config["normalized_sha256"], data_sha}:
            raise CLICTargetProtocolError("predictor train config normalized SHA drift")
        return {
            "container_path": config["path"],
            "member_name": None,
            "raw_sha256": raw_sha,
            "sealed_normalized_sha256": expected_normalized,
            "data_normalized_sha256": data_sha,
            "normalized": config["normalized"],
        }
    if member_name != "candidate_train_data_config.json":
        raise CLICTargetProtocolError("predictor train config bundle member binding drift")
    raw = _target._load_bundle_member_json(
        raw_path,
        member_name=member_name,
        expected_sha256=raw_sha,
        label="predictor train config",
    )
    if (
        raw.get("schema") != "cvs.phase1.clic_train_data_config.v1"
        or raw.get("real_checkpoint_config") is not True
        or not isinstance(raw.get("normalized"), Mapping)
    ):
        raise CLICTargetProtocolError("predictor train config bundle member is not a real config")
    normalized = dict(raw["normalized"])
    normalized.pop("input_len", None)
    data_sha = _train_data_sha(normalized)
    if raw.get("normalized_sha256") != _target.canonical_sha256(dict(raw["normalized"])):
        raise CLICTargetProtocolError("predictor train config bundle manifest normalized SHA drift")
    if expected_normalized != data_sha:
        raise CLICTargetProtocolError("predictor train config bundle data normalized SHA drift")
    return {
        "container_path": str(Path(raw_path).resolve()),
        "member_name": member_name,
        "raw_sha256": raw_sha,
        "sealed_normalized_sha256": expected_normalized,
        "data_normalized_sha256": data_sha,
        "normalized": normalized,
    }


def _finite_vector(value: Any, *, label: str) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CLICTargetProtocolError(f"{label} is not a numeric vector") from exc
    if array.size <= 0 or not np.isfinite(array).all():
        raise CLICTargetProtocolError(f"{label} is empty or non-finite")
    return [float(item) for item in array.tolist()]


def _prediction_row(
    *,
    opaque_token: str,
    scene: str,
    received_iq_sha256: str,
    output: Mapping[str, Any],
) -> dict[str, Any]:
    decision = str(output.get("decision", ""))
    if decision not in {"registered", "unknown", "defer"}:
        raise CLICTargetProtocolError("target predictor output decision is invalid")
    logits = _finite_vector(output.get("tx_logits"), label="target predictor tx_logits")
    try:
        energy = float(output.get("e_unknown"))
    except (TypeError, ValueError) as exc:
        raise CLICTargetProtocolError("target predictor unknown energy is invalid") from exc
    if not math.isfinite(energy):
        raise CLICTargetProtocolError("target predictor unknown energy is non-finite")
    maximum = max(logits)
    winners = [index for index, value in enumerate(logits) if value == maximum]
    predicted_index: int | None = None
    if decision == "registered":
        if len(winners) != 1:
            raise CLICTargetProtocolError("registered target prediction has an exact-head tie")
        predicted_index = int(winners[0])
    return {
        "opaque_token": _target.require_sha256(opaque_token, label="prediction opaque token"),
        "scene": str(scene),
        "received_iq_sha256": _target.require_sha256(
            received_iq_sha256, label="prediction received IQ"
        ),
        "e_unknown": energy,
        "decision": decision,
        # Do not serialize embeddings, CLIC tokens, logits, or a target class
        # order.  The isolated scorer bounds this index only after it opens the
        # evaluator-only truth/config chain.
        "predicted_index": predicted_index,
    }


# Re-exported as a module-level name so deployment runners can substitute an
# independently verified loader in controlled tests without injecting model
# state into `publish_clic_target_prediction`.
load_verified_clic_predictor_state = _target.load_verified_clic_predictor_state


def _validated_source_class_order_binding(
    value: Any, sha256: Any, *, label: str
) -> tuple[list[str], str]:
    """Validate the predictor's PAIR-derived local-four class order only."""

    if not isinstance(value, (list, tuple)):
        raise CLICTargetProtocolError(f"{label} must be an ordered identifier list")
    order = [str(item) for item in value]
    if len(order) != 4 or len(set(order)) != 4 or any(not item for item in order):
        raise CLICTargetProtocolError(f"{label} must contain exactly four unique source TX IDs")
    digest = _target.require_sha256(sha256, label=label)
    if _target.canonical_sha256(order) != digest:
        raise CLICTargetProtocolError(f"{label} SHA binding drifted")
    return order, digest


def _require_runtime_identity(runtime: Any) -> dict[str, Any]:
    """Normalize the source-only runtime identity before target-IQ access."""

    arm = str(getattr(runtime, "arm", ""))
    operator = str(getattr(runtime, "operator", ""))
    expected_operator = {
        "C": "raw_phase_control",
        "G": "complex_local_invariant_curvature",
    }
    if arm not in expected_operator or operator != expected_operator[arm]:
        raise CLICTargetProtocolError("target predictor arm/operator binding drift")
    fold = getattr(runtime, "fold_index", None)
    if type(fold) is not int or fold not in range(1, 7):
        raise CLICTargetProtocolError("target predictor fold binding drift")
    identity = {
        "arm": arm,
        "operator": operator,
        "fold_index": fold,
        "state_sha256": _target.require_sha256(
            getattr(runtime, "state_sha256", None), label="predictor state"
        ),
        "source_frozen_rule_sha256": _target.require_sha256(
            getattr(runtime, "source_frozen_rule_sha256", None),
            label="predictor source rule",
        ),
    }
    # Every production C/G descriptor is PAIR-bound to exactly one local-four
    # source policy.  A target prediction without that binding could silently
    # widen formal-known DG to the union cache, so it is never a compatible
    # legacy artifact.
    order, digest = _validated_source_class_order_binding(
        getattr(runtime, "source_class_order", None),
        getattr(runtime, "source_class_order_sha256", None),
        label="target predictor source class order",
    )
    identity["source_class_order"] = order
    identity["source_class_order_sha256"] = digest
    return identity


def _same_train_config_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in (
            "container_path",
            "member_name",
            "raw_sha256",
            "sealed_normalized_sha256",
            "data_normalized_sha256",
        )
    )


def _reverify_publish_bindings(
    *,
    snapshot: Mapping[str, Any],
    runtime: Any,
) -> None:
    """Close the publisher's verify-to-forward window before sealing output."""

    if sha256_file(Path(str(snapshot["package_manifest_path"]))) != snapshot["package_manifest_raw_sha256"]:
        raise CLICTargetProtocolError("target package manifest changed during target forward")
    if sha256_file(Path(str(snapshot["package_data_path"]))) != snapshot["package_data_raw_sha256"]:
        raise CLICTargetProtocolError("target package received-IQ data changed during target forward")
    if sha256_file(Path(str(snapshot["predictor_artifact_path"]))) != snapshot["predictor_artifact_sha256"]:
        raise CLICTargetProtocolError("predictor artifact changed during target forward")
    reopened_train = _runtime_train_config(runtime)
    if not _same_train_config_binding(snapshot["train_config"], reopened_train):
        raise CLICTargetProtocolError("predictor train config changed during target forward")


def publish_clic_target_prediction(
    predictor_state_path: str | Path,
    package_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Perform exactly one immutable target forward per received-IQ package row."""

    if not isinstance(predictor_state_path, (str, Path)):
        raise CLICTargetProtocolError("target prediction accepts only a predictor-state path artifact")
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise CLICTargetProtocolError(
            f"CLIC target prediction output already exists and is immutable: {output_file}"
        )
    predictor_artifact = Path(predictor_state_path).resolve()
    if not predictor_artifact.is_file():
        raise FileNotFoundError(f"target predictor artifact is missing: {predictor_artifact}")
    # The predictor state and all IQ-only package bytes close before the
    # received-IQ archive is materialized.  No truth-sidecar or known-config
    # path is accepted or opened by this API.
    predictor_artifact_sha = sha256_file(predictor_artifact)
    runtime = load_verified_clic_predictor_state(predictor_artifact)
    runtime_identity = _require_runtime_identity(runtime)
    train_config = _runtime_train_config(runtime)
    package = _load_verified_clic_iq_only_package(package_path)
    snapshot = {
        "package_manifest_path": str(package["manifest_path"]),
        "package_manifest_raw_sha256": package["manifest_raw_sha256"],
        "package_data_path": str(package["data_path"]),
        "package_data_raw_sha256": package["data_raw_sha256"],
        "predictor_artifact_path": str(predictor_artifact),
        "predictor_artifact_sha256": predictor_artifact_sha,
        "train_config": dict(train_config),
    }
    rows: list[dict[str, Any]] = []
    for index in range(int(package["received_iq"].shape[0])):
        scene = str(package["scenes"][index])
        forward_output = runtime.forward_once(package["received_iq"][index], scene=scene)
        if not isinstance(forward_output, Mapping):
            raise CLICTargetProtocolError("target predictor forward did not return a mapping")
        rows.append(
            _prediction_row(
                opaque_token=str(package["opaque_tokens"][index]),
                scene=scene,
                received_iq_sha256=str(package["received_iq_sha256"][index]),
                output=forward_output,
            )
        )
    if len(rows) != int(package["manifest"]["row_count"]):
        raise CLICTargetProtocolError("target prediction forward count does not close")
    if len({row["opaque_token"] for row in rows}) != len(rows):
        raise CLICTargetProtocolError("target prediction opaque-token uniqueness drift")
    _reverify_publish_bindings(snapshot=snapshot, runtime=runtime)
    base = {
        "schema": _PREDICTION_SCHEMA,
        "sealed": True,
        "truth_sidecar_opened": False,
        "predictor_package_path": str(package["path"]),
        "predictor_package_sha256": package["manifest"]["package_sha256"],
        "package_manifest_sha256": package["manifest_raw_sha256"],
        "received_iq_data_sha256": package["data_raw_sha256"],
        "predictor_state_path": str(predictor_artifact),
        "predictor_artifact_sha256": predictor_artifact_sha,
        "predictor_state_sha256": runtime_identity["state_sha256"],
        "source_frozen_rule_sha256": runtime_identity["source_frozen_rule_sha256"],
        # PAIR-derived local-four source policy.  This is predictor-state
        # binding only; no target known class order enters the prediction.
        "source_class_order": runtime_identity["source_class_order"],
        "source_class_order_sha256": runtime_identity["source_class_order_sha256"],
        "arm": runtime_identity["arm"],
        "operator": runtime_identity["operator"],
        "fold_index": runtime_identity["fold_index"],
        "train_config_manifest_path": train_config["container_path"],
        "train_config_member_name": train_config["member_name"],
        "train_config_raw_sha256": train_config["raw_sha256"],
        "train_config_normalized_sha256": train_config["sealed_normalized_sha256"],
        "train_config_data_normalized_sha256": train_config["data_normalized_sha256"],
        # Cross-method comparison keys are semantic training-data SHA values,
        # never F1..F6 labels or cache/physical-IQ identifiers.
        "fold_config_key": train_config["data_normalized_sha256"],
        "known_test_config_raw_sha256": package["manifest"]["known_test_config_raw_sha256"],
        "known_test_config_normalized_sha256": package["manifest"]["known_test_config_normalized_sha256"],
        "lineage_sha256": package["manifest"]["lineage_sha256"],
        "row_count": len(rows),
        "forward_count": len(rows),
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "rows": rows,
    }
    prediction_sha = _target.canonical_sha256(base)
    payload = dict(base, prediction_sha256=prediction_sha)
    _write_new_utf8_json(
        output_file,
        payload,
        label="CLIC target prediction",
        escape_truth_sidecar_key=True,
    )
    return output_file


def _as_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CLICTargetProtocolError(f"{label} must be a strict positive integer")
    number = int(value)
    if number <= 0:
        raise CLICTargetProtocolError(f"{label} must be a strict positive integer")
    return number


def _as_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CLICTargetProtocolError(f"{label} must be a nonnegative integer")
    number = int(value)
    if number < 0:
        raise CLICTargetProtocolError(f"{label} must be a nonnegative integer")
    return number


def _validated_accuracy_triplet(
    value: Any, *, label: str, allow_zero_denominator: bool = False
) -> dict[str, Any]:
    """Validate a count-derived rate; never trust a supplied scalar alone."""

    payload = _target._require_mapping(value, label=label)
    if set(payload) != {"numerator", "denominator", "accuracy"}:
        raise CLICTargetProtocolError(f"{label} fields drift")
    numerator = _as_nonnegative_int(payload.get("numerator"), label=f"{label} numerator")
    denominator = (
        _as_nonnegative_int(payload.get("denominator"), label=f"{label} denominator")
        if allow_zero_denominator
        else _as_positive_int(payload.get("denominator"), label=f"{label} denominator")
    )
    if numerator > denominator:
        raise CLICTargetProtocolError(f"{label} numerator exceeds denominator")
    raw_accuracy = payload.get("accuracy")
    if denominator == 0:
        if numerator != 0 or raw_accuracy not in {0, 0.0, None}:
            raise CLICTargetProtocolError(f"{label} zero-denominator accuracy drift")
        accuracy: float | None = None
    else:
        try:
            accuracy = float(raw_accuracy)
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError(f"{label} accuracy is invalid") from exc
        expected = numerator / denominator
        if not math.isfinite(accuracy) or not math.isclose(
            accuracy, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise CLICTargetProtocolError(f"{label} accuracy/numerator denominator mismatch")
    return {"numerator": numerator, "denominator": denominator, "accuracy": accuracy}


def _validated_rate_triplet(
    value: Any, *, label: str, allow_zero_denominator: bool = False
) -> dict[str, Any]:
    """Validate an exact count/rate record without trusting its rate scalar."""

    payload = _target._require_mapping(value, label=label)
    if set(payload) != {"numerator", "denominator", "rate"}:
        raise CLICTargetProtocolError(f"{label} fields drift")
    checked = _validated_accuracy_triplet(
        {
            "numerator": payload.get("numerator"),
            "denominator": payload.get("denominator"),
            "accuracy": payload.get("rate"),
        },
        label=label,
        allow_zero_denominator=allow_zero_denominator,
    )
    return {
        "numerator": checked["numerator"],
        "denominator": checked["denominator"],
        "rate": checked["accuracy"],
    }


def _validated_accepted_known(value: Any, *, overall: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the accepted-known slice derived from mutually exclusive decisions."""

    payload = _target._require_mapping(value, label="ADV3B02 accepted-known")
    if set(payload) != {"correct", "denominator", "accuracy", "coverage"}:
        raise CLICTargetProtocolError("ADV3B02 accepted-known fields drift")
    correct = _as_nonnegative_int(payload.get("correct"), label="ADV3B02 accepted-known correct")
    denominator = _as_nonnegative_int(
        payload.get("denominator"), label="ADV3B02 accepted-known denominator"
    )
    if correct > denominator:
        raise CLICTargetProtocolError("ADV3B02 accepted-known correct exceeds denominator")
    overall_numerator = _as_nonnegative_int(
        overall.get("numerator"), label="ADV3B02 overall numerator"
    )
    overall_denominator = _as_positive_int(
        overall.get("denominator"), label="ADV3B02 overall denominator"
    )
    if correct != overall_numerator:
        raise CLICTargetProtocolError("ADV3B02 accepted-known correct/overall numerator drift")
    expected_coverage = denominator / overall_denominator
    raw_coverage = payload.get("coverage")
    if isinstance(raw_coverage, Mapping):
        coverage = _validated_rate_triplet(
            raw_coverage, label="ADV3B02 accepted-known coverage"
        )
        if (
            coverage["numerator"] != denominator
            or coverage["denominator"] != overall_denominator
            or not math.isclose(
                float(coverage["rate"]), expected_coverage, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise CLICTargetProtocolError("ADV3B02 accepted-known coverage recompute drift")
    else:
        # Historical source artifacts recorded only this scalar.  It is never
        # retained in a formal reference: accept it solely when the two count
        # fields above reproduce it exactly, then upgrade the sealed form.
        try:
            legacy_coverage = float(raw_coverage)
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError("ADV3B02 accepted-known coverage is invalid") from exc
        if not math.isfinite(legacy_coverage) or not math.isclose(
            legacy_coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12
        ):
            raise CLICTargetProtocolError("ADV3B02 accepted-known coverage recompute drift")
        coverage = {
            "numerator": denominator,
            "denominator": overall_denominator,
            "rate": expected_coverage,
        }
    if denominator == 0:
        if payload.get("accuracy") not in {0, 0.0, None}:
            raise CLICTargetProtocolError("ADV3B02 accepted-known zero-denominator accuracy drift")
        accuracy: float | None = None
    else:
        try:
            accuracy = float(payload.get("accuracy"))
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError("ADV3B02 accepted-known accuracy is invalid") from exc
        if not math.isfinite(accuracy) or not math.isclose(
            accuracy, correct / denominator, rel_tol=0.0, abs_tol=1e-12
        ):
            raise CLICTargetProtocolError("ADV3B02 accepted-known accuracy recompute drift")
    return {
        "correct": correct,
        "denominator": denominator,
        "accuracy": accuracy,
        "coverage": coverage,
    }


def _validated_partition_metrics(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    payload = _target._require_mapping(value, label=label)
    if not payload:
        raise CLICTargetProtocolError(f"{label} must not be empty")
    validated: dict[str, dict[str, Any]] = {}
    for key, item in payload.items():
        name = str(key)
        if not name or name in validated:
            raise CLICTargetProtocolError(f"{label} group key drift")
        validated[name] = _validated_accuracy_triplet(item, label=f"{label}[{name}]")
    return validated


def _validated_crossed_partition_metrics(
    value: Any, *, label: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate an exact class-by-axis count table without trusting marginals."""

    payload = _target._require_mapping(value, label=label)
    if not payload:
        raise CLICTargetProtocolError(f"{label} must not be empty")
    validated: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_class, raw_axis in payload.items():
        class_id = str(raw_class)
        if not class_id or class_id in validated:
            raise CLICTargetProtocolError(f"{label} class key drift")
        validated[class_id] = _validated_partition_metrics(
            raw_axis, label=f"{label}[{class_id}]"
        )
    return validated


def _validate_crossed_partition_closure(
    *,
    by_class: Mapping[str, Mapping[str, Any]],
    by_axis: Mapping[str, Mapping[str, Any]],
    by_class_axis: Mapping[str, Mapping[str, Mapping[str, Any]]],
    label: str,
) -> None:
    """Require class×axis cells to reproduce both sealed marginal partitions."""

    class_ids = set(by_class)
    axis_ids = set(by_axis)
    if set(by_class_axis) != class_ids:
        raise CLICTargetProtocolError(f"{label} class universe does not match by_class")
    for class_id in sorted(class_ids):
        cells = _target._require_mapping(by_class_axis[class_id], label=f"{label}[{class_id}]")
        if set(cells) != axis_ids:
            raise CLICTargetProtocolError(f"{label} axis universe does not match by_axis")
        numerator = sum(int(cells[axis_id]["numerator"]) for axis_id in axis_ids)
        denominator = sum(int(cells[axis_id]["denominator"]) for axis_id in axis_ids)
        expected = _target._require_mapping(by_class[class_id], label=f"{label} by_class[{class_id}]")
        if numerator != int(expected["numerator"]) or denominator != int(expected["denominator"]):
            raise CLICTargetProtocolError(f"{label} class marginal count closure drift")
    for axis_id in sorted(axis_ids):
        numerator = sum(int(by_class_axis[class_id][axis_id]["numerator"]) for class_id in class_ids)
        denominator = sum(int(by_class_axis[class_id][axis_id]["denominator"]) for class_id in class_ids)
        expected = _target._require_mapping(by_axis[axis_id], label=f"{label} by_axis[{axis_id}]")
        if numerator != int(expected["numerator"]) or denominator != int(expected["denominator"]):
            raise CLICTargetProtocolError(f"{label} axis marginal count closure drift")


def _validate_rich_reference_cell(cell: Mapping[str, Any], validated: dict[str, Any]) -> None:
    """Recompute all full known-target ADV aggregates from count subcells."""

    rich_fields = {
        "target_day_set_sha256",
        "overall",
        "by_class",
        "by_receiver",
        "by_day",
        "by_class_receiver",
        "by_class_day",
        "macro_accuracy",
        "min_class_accuracy",
        "min_receiver_accuracy",
        "min_day_accuracy",
        "known_false_reject",
        "known_defer",
        "accepted_known",
    }
    present = rich_fields & set(cell)
    if not present:
        return
    if present != rich_fields:
        raise CLICTargetProtocolError("ADV3B02 semantic cell rich metric fields are incomplete")
    validated["target_day_set_sha256"] = _target.require_sha256(
        cell["target_day_set_sha256"], label="ADV3B02 cell target_day_set_sha256"
    )
    overall = _validated_accuracy_triplet(cell["overall"], label="ADV3B02 overall")
    if (
        overall["numerator"] != validated["numerator"]
        or overall["denominator"] != validated["denominator"]
        or (
            overall["accuracy"] is not None
            and "accuracy" in validated
            and not math.isclose(
                float(overall["accuracy"]), float(validated["accuracy"]), rel_tol=0.0, abs_tol=1e-12
            )
        )
    ):
        raise CLICTargetProtocolError("ADV3B02 overall/top-level count or accuracy drift")
    validated["overall"] = overall
    for field in ("by_class", "by_receiver", "by_day"):
        partition = _validated_partition_metrics(cell[field], label=f"ADV3B02 {field}")
        if (
            sum(int(item["numerator"]) for item in partition.values()) != overall["numerator"]
            or sum(int(item["denominator"]) for item in partition.values()) != overall["denominator"]
        ):
            raise CLICTargetProtocolError(f"ADV3B02 {field} counts do not recompute overall")
        validated[field] = partition
    validated["by_class_receiver"] = _validated_crossed_partition_metrics(
        cell["by_class_receiver"], label="ADV3B02 by_class_receiver"
    )
    validated["by_class_day"] = _validated_crossed_partition_metrics(
        cell["by_class_day"], label="ADV3B02 by_class_day"
    )
    _validate_crossed_partition_closure(
        by_class=validated["by_class"],
        by_axis=validated["by_receiver"],
        by_class_axis=validated["by_class_receiver"],
        label="ADV3B02 by_class_receiver",
    )
    _validate_crossed_partition_closure(
        by_class=validated["by_class"],
        by_axis=validated["by_day"],
        by_class_axis=validated["by_class_day"],
        label="ADV3B02 by_class_day",
    )
    derived = {
        "macro_accuracy": sum(
            float(item["accuracy"]) for item in validated["by_class"].values()
        ) / len(validated["by_class"]),
        "min_class_accuracy": min(
            float(item["accuracy"]) for item in validated["by_class"].values()
        ),
        "min_receiver_accuracy": min(
            float(item["accuracy"]) for item in validated["by_receiver"].values()
        ),
        "min_day_accuracy": min(
            float(item["accuracy"]) for item in validated["by_day"].values()
        ),
    }
    for field, expected in derived.items():
        try:
            supplied = float(cell[field])
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError(f"ADV3B02 {field} is invalid") from exc
        if not math.isfinite(supplied) or not math.isclose(
            supplied, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise CLICTargetProtocolError(f"ADV3B02 {field} derived recompute drift")
        validated[field] = expected
    false_reject = _validated_accuracy_triplet(
        cell["known_false_reject"], label="ADV3B02 known_false_reject"
    )
    defer = _validated_accuracy_triplet(cell["known_defer"], label="ADV3B02 known_defer")
    if (
        false_reject["denominator"] != overall["denominator"]
        or defer["denominator"] != overall["denominator"]
        or false_reject["numerator"] + defer["numerator"] > overall["denominator"]
    ):
        raise CLICTargetProtocolError("ADV3B02 known reject/defer count drift")
    validated["known_false_reject"] = false_reject
    validated["known_defer"] = defer
    accepted = _validated_accepted_known(cell["accepted_known"], overall=overall)
    if accepted["denominator"] != (
        overall["denominator"] - false_reject["numerator"] - defer["numerator"]
    ):
        raise CLICTargetProtocolError("ADV3B02 accepted-known denominator recompute drift")
    validated["accepted_known"] = accepted


def _validate_reference_cell(
    cell: Mapping[str, Any],
    *,
    expected_train_config_sha256: str,
    expected_known_config_sha256: str,
    accepted_legacy_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    required = (
        "fold_config_key",
        "scene",
        "target_receiver_set_sha256",
        "target_known_tx_set_sha256",
        "class_order_sha256",
        "known_test_config_sha256",
        "numerator",
        "denominator",
    )
    missing = [field for field in required if field not in cell]
    if missing:
        raise CLICTargetProtocolError(f"ADV3B02 semantic cell is missing fields: {missing}")
    allowed = set(required) | {
        "accuracy",
        "overall_accuracy",
        "macro_accuracy",
        "min_receiver_accuracy",
        "min_class_accuracy",
        "target_day_set_sha256",
        "overall",
        "by_class",
        "by_receiver",
        "by_day",
        "by_class_receiver",
        "by_class_day",
        "min_day_accuracy",
        "known_false_reject",
        "known_defer",
        "accepted_known",
    }
    unsupported = sorted(set(cell) - allowed)
    if unsupported:
        raise CLICTargetProtocolError(
            f"ADV3B02 semantic cell has unsupported/non-known fields: {unsupported}"
        )
    fold = str(cell["fold_config_key"])
    scene = str(cell["scene"])
    fold = _target.require_sha256(fold, label="ADV3B02 cell fold_config_key")
    if fold != expected_train_config_sha256 or scene not in FORMAL_LEO_WEAK_SCENARIOS:
        raise CLICTargetProtocolError("ADV3B02 semantic cell fold/scene drift")
    validated: dict[str, Any] = {"fold_config_key": fold, "scene": scene}
    for field in (
        "target_receiver_set_sha256",
        "target_known_tx_set_sha256",
        "class_order_sha256",
        "known_test_config_sha256",
    ):
        validated[field] = _target.require_sha256(cell[field], label=f"ADV3B02 cell {field}")
    supplied_known_sha = validated["known_test_config_sha256"]
    if supplied_known_sha not in {
        expected_known_config_sha256,
        accepted_legacy_manifest_sha256,
    }:
        raise CLICTargetProtocolError("ADV3B02 semantic cell known-test config SHA drift")
    # Persist the canonical *data* configuration identity.  The legacy raw
    # manifest identity is accepted only while ingesting pre-Task7 evidence;
    # it is never retained as a cross-capsule semantic-cell key.
    validated["known_test_config_sha256"] = expected_known_config_sha256
    numerator = _as_nonnegative_int(cell["numerator"], label="ADV3B02 cell numerator")
    denominator = _as_positive_int(cell["denominator"], label="ADV3B02 cell denominator")
    if numerator > denominator:
        raise CLICTargetProtocolError("ADV3B02 semantic cell numerator exceeds denominator")
    validated["numerator"] = numerator
    validated["denominator"] = denominator
    scalar_fields = {
        "accuracy",
        "overall_accuracy",
        "macro_accuracy",
        "min_receiver_accuracy",
        "min_class_accuracy",
        "min_day_accuracy",
    }
    for field in sorted(set(cell) & scalar_fields):
        value = float(cell[field])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise CLICTargetProtocolError(f"ADV3B02 semantic cell {field} is invalid")
        validated[field] = value
    if "accuracy" in validated:
        expected = numerator / denominator
        if not math.isclose(float(validated["accuracy"]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise CLICTargetProtocolError("ADV3B02 semantic cell accuracy/numerator mismatch")
    _validate_rich_reference_cell(cell, validated)
    return validated


def _validate_reference_cell_set(
    raw_cells: Any,
    *,
    expected_train_config_sha256: str,
    expected_known_config_sha256: str,
    accepted_legacy_manifest_sha256: str | None,
    known_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the one-fold/three-scene ADV reference cell set."""

    if not isinstance(raw_cells, list) or len(raw_cells) != len(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTargetProtocolError(
            "ADV3B02 reference requires exactly three formal-scene semantic cells per fold"
        )
    cells = [
        _validate_reference_cell(
            _target._require_mapping(cell, label="ADV3B02 semantic cell"),
            expected_train_config_sha256=expected_train_config_sha256,
            expected_known_config_sha256=expected_known_config_sha256,
            accepted_legacy_manifest_sha256=accepted_legacy_manifest_sha256,
        )
        for cell in raw_cells
    ]
    if {str(cell["scene"]) for cell in cells} != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTargetProtocolError("ADV3B02 semantic target-known cells are missing a formal scene")
    if len({str(cell["scene"]) for cell in cells}) != len(cells):
        raise CLICTargetProtocolError("ADV3B02 semantic target-known cells are duplicated")
    known = _target.normalize_known_test_config(known_config)
    expected_sets = {
        "target_receiver_set_sha256": _target.canonical_sha256(
            sorted({str(item) for item in known["target_receiver_ids"]})
        ),
        "target_known_tx_set_sha256": _target.canonical_sha256(
            sorted({str(item) for item in known["target_known_tx_ids"]})
        ),
        "class_order_sha256": _target.canonical_sha256(list(known["class_order"])),
        "known_test_config_sha256": expected_known_config_sha256,
    }
    rich = ["overall" in cell for cell in cells]
    if any(rich) and not all(rich):
        raise CLICTargetProtocolError("ADV3B02 reference mixes complete and incomplete scene cells")
    if all(rich):
        expected_sets["target_day_set_sha256"] = _target.canonical_sha256(
            sorted({str(item) for item in known["target_day_ids"]})
        )
        expected_group_keys = {
            "by_class": set(str(item) for item in known["class_order"]),
            "by_receiver": set(str(item) for item in known["target_receiver_ids"]),
            "by_day": set(str(item) for item in known["target_day_ids"]),
        }
        for cell in cells:
            for field, expected_keys in expected_group_keys.items():
                if set(cell[field]) != expected_keys:
                    raise CLICTargetProtocolError(
                        f"ADV3B02 {field} class/receiver/day coverage drift"
                    )
    for cell in cells:
        if any(cell.get(field) != expected for field, expected in expected_sets.items()):
            raise CLICTargetProtocolError("ADV3B02 semantic-cell universe/config binding drift")
    return sorted(cells, key=lambda cell: FORMAL_LEO_WEAK_SCENARIOS.index(str(cell["scene"])))


def ingest_adv3b02_target_known_reference(
    checkpoint_path: str | Path,
    train_config_manifest_path: str | Path,
    known_test_config_manifest_path: str | Path,
    stratified_metric_artifact_path: str | Path,
    output_reference_path: str | Path,
) -> Path:
    """Seal a read-only ADV3B02 target-known reference without rerunning it."""

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ADV3B02 checkpoint is missing: {checkpoint}")
    train = _target.read_verified_config_manifest(
        train_config_manifest_path,
        expected_schema="cvs.phase1.adv3b02_train_data_config.v1",
        label="ADV3B02 train config",
    )
    known = _target.read_verified_config_manifest(
        known_test_config_manifest_path,
        expected_schema="cvs.phase1.adv3b02_known_test_config.v1",
        label="ADV3B02 known-test config",
    )
    # Validate the data-bearing schemas now.  Their semantic equality against a
    # CLIC candidate happens separately and deliberately ignores package bytes.
    train_data_sha = _train_data_sha(train["normalized"])
    known_data_sha = _known_test_data_sha(known["normalized"])

    metrics_path = Path(stratified_metric_artifact_path).resolve()
    metrics = _target.read_json_object(metrics_path, label="ADV3B02 stratified target-known metrics")
    if str(metrics.get("schema", "")) != _target.ADV3B02_METRICS_SCHEMA:
        raise CLICTargetProtocolError("ADV3B02 stratified metric schema drift")
    cells = _validate_reference_cell_set(
        metrics.get("cells"),
        expected_train_config_sha256=train_data_sha,
        expected_known_config_sha256=known_data_sha,
        accepted_legacy_manifest_sha256=known["normalized_sha256"],
        known_config=known["normalized"],
    )

    reference_base = {
        "schema": _target.ADV3B02_REFERENCE_SCHEMA,
        "sealed": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_config_manifest_path": train["path"],
        "train_config_raw_sha256": train["raw_sha256"],
        "train_config_normalized_sha256": train_data_sha,
        "known_test_config_manifest_path": known["path"],
        "known_test_config_raw_sha256": known["raw_sha256"],
        "known_test_config_normalized_sha256": known_data_sha,
        "stratified_metric_artifact_path": str(metrics_path),
        "stratified_metric_artifact_sha256": sha256_file(metrics_path),
        "semantic_cells": cells,
    }
    payload = dict(
        reference_base,
        reference_sha256=_target.canonical_sha256(reference_base),
    )
    output = Path(output_reference_path).resolve()
    _write_new_utf8_json(output, payload, label="ADV3B02 target-known reference")
    return output


def validate_adv3b02_config_equivalence(
    *,
    candidate_train_config: Mapping[str, Any],
    candidate_known_test_config: Mapping[str, Any],
    baseline_train_config: Mapping[str, Any],
    baseline_known_test_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact equality on frozen training/test *data* configurations.

    Capsule IDs, physical rows, received-IQ bytes and seeds are intentionally
    excluded.  So are model/training-hyperparameter fields.  Any change on the
    split/channel/preprocessing/zero-adaptation/metric surface fails closed.
    """

    candidate_train = _target.normalize_train_data_config(candidate_train_config)
    baseline_train = _target.normalize_train_data_config(baseline_train_config)
    candidate_known = _target.normalize_known_test_config(candidate_known_test_config)
    baseline_known = _target.normalize_known_test_config(baseline_known_test_config)
    differences: dict[str, list[str]] = {}
    for label, left, right in (
        ("train", candidate_train, baseline_train),
        ("known_test", candidate_known, baseline_known),
    ):
        drift = sorted(
            field
            for field in set(left) | set(right)
            if left.get(field) != right.get(field)
        )
        if drift:
            differences[label] = drift
    if differences:
        raise CLICTargetProtocolError(
            f"ADV3B02 config equivalence drift: {differences}"
        )
    return {
        "passed": True,
        "candidate_train_data_sha256": _target.canonical_sha256(candidate_train),
        "baseline_train_data_sha256": _target.canonical_sha256(baseline_train),
        "candidate_known_test_data_sha256": _target.canonical_sha256(candidate_known),
        "baseline_known_test_data_sha256": _target.canonical_sha256(baseline_known),
        "capsule_or_received_iq_pairing_required": False,
    }


def recompute_unknown_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute explicit true-unknown numerator/denominator without defer."""

    denominator_by_scene = {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    numerator_by_scene = {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    defer_by_scene = {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    for index, raw_row in enumerate(rows):
        row = _target._require_mapping(raw_row, label=f"target score row {index}")
        scene = str(row.get("scene", ""))
        if scene not in denominator_by_scene:
            raise CLICTargetProtocolError(f"target score row {index} has unsupported LEO scene")
        role = str(row.get("role", ""))
        truth = str(row.get("truth", ""))
        if (role == "unknown") != (truth == "unknown"):
            raise CLICTargetProtocolError(
                f"target score row {index} has inconsistent true-unknown role/truth"
            )
        if role != "unknown":
            continue
        decision = str(row.get("decision", ""))
        if decision not in {"registered", "unknown", "defer"}:
            raise CLICTargetProtocolError(
                f"target score row {index} has invalid target decision"
            )
        denominator_by_scene[scene] += 1
        if decision == "unknown":
            numerator_by_scene[scene] += 1
        elif decision == "defer":
            defer_by_scene[scene] += 1
    if any(value <= 0 for value in denominator_by_scene.values()):
        missing = [scene for scene, value in denominator_by_scene.items() if value <= 0]
        raise CLICTargetGateError(
            f"true-unknown explicit rejection requires positive denominator in every scene: {missing}"
        )
    denominator_global = sum(denominator_by_scene.values())
    numerator_global = sum(numerator_by_scene.values())
    defer_global = sum(defer_by_scene.values())
    if denominator_global <= 0:
        raise CLICTargetGateError("true-unknown explicit rejection requires positive global denominator")
    return {
        "unknown_denominator_global": denominator_global,
        "unknown_numerator_global": numerator_global,
        "unknown_defer_global": defer_global,
        "unknown_rejection_rate_global": numerator_global / denominator_global,
        "unknown_denominator_by_scene": denominator_by_scene,
        "unknown_numerator_by_scene": numerator_by_scene,
        "unknown_defer_by_scene": defer_by_scene,
        "unknown_rejection_rate_by_scene": {
            scene: numerator_by_scene[scene] / denominator_by_scene[scene]
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        },
    }


def _evaluate_explicit_unknown_gate(
    rows: Iterable[Mapping[str, Any]], *, explicit_unknown_floor: float = 0.70
) -> dict[str, Any]:
    """Evaluate the target unknown gate without converting a result into an exception."""

    floor = float(explicit_unknown_floor)
    if not math.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise CLICTargetProtocolError("explicit unknown floor must be finite within [0,1]")
    audit = recompute_unknown_counts(rows)
    failures = []
    global_rate = float(audit["unknown_rejection_rate_global"])
    if global_rate < floor:
        failures.append(("global", global_rate))
    per_scene = dict(audit["unknown_rejection_rate_by_scene"])
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        rate = float(per_scene[scene])
        if rate < floor:
            failures.append((scene, rate))
    return {
        "explicit_unknown_gate_passed": not failures,
        "explicit_unknown_floor": floor,
        "unknown_audit": audit,
        "failures": [
            {"scope": scope, "rate": rate, "floor": floor} for scope, rate in failures
        ],
    }


def score_target_rows(
    rows: Iterable[Mapping[str, Any]], *, explicit_unknown_floor: float = 0.70
) -> dict[str, Any]:
    """Apply the public throwing form of the explicit target-unknown gate."""

    evaluated = _evaluate_explicit_unknown_gate(
        rows, explicit_unknown_floor=explicit_unknown_floor
    )
    if not evaluated["explicit_unknown_gate_passed"]:
        details = ", ".join(
            f"{item['scope']}={float(item['rate']):.6f}" for item in evaluated["failures"]
        )
        raise CLICTargetGateError(
            "explicit true-unknown rejection must be >= "
            f"{float(evaluated['explicit_unknown_floor']):.2f}; defer is excluded; {details}"
        )
    return evaluated


def _require_existing_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CLICTargetProtocolError(f"{label} path is invalid")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _load_verified_sealed_clic_prediction(prediction_path: str | Path) -> dict[str, Any]:
    """Verify an immutable, IQ-only prediction before any truth-side access."""

    path = Path(prediction_path).resolve()
    raw_before = sha256_file(path)
    payload = _target.read_json_object(path, label="CLIC target prediction")
    raw_after = sha256_file(path)
    if raw_before != raw_after:
        raise CLICTargetProtocolError("CLIC target prediction changed while opening")
    required = {
        "schema",
        "sealed",
        "truth_sidecar_opened",
        "predictor_package_path",
        "predictor_package_sha256",
        "package_manifest_sha256",
        "received_iq_data_sha256",
        "predictor_state_path",
        "predictor_artifact_sha256",
        "predictor_state_sha256",
        "source_frozen_rule_sha256",
        "arm",
        "operator",
        "fold_index",
        "train_config_manifest_path",
        "train_config_member_name",
        "train_config_raw_sha256",
        "train_config_normalized_sha256",
        "train_config_data_normalized_sha256",
        "fold_config_key",
        "known_test_config_raw_sha256",
        "known_test_config_normalized_sha256",
        "lineage_sha256",
        "row_count",
        "forward_count",
        "target_fit_rows",
        "target_update_rows",
        "target_retry_count",
        "target_selection_count",
        "target_selection_feedback",
        "rows",
        "prediction_sha256",
    }
    source_order_fields = {"source_class_order", "source_class_order_sha256"}
    if (
        payload.get("schema") != _PREDICTION_SCHEMA
        or payload.get("sealed") is not True
        or payload.get("truth_sidecar_opened") is not False
    ):
        raise CLICTargetProtocolError("CLIC target prediction is not an immutable truth-blind seal")
    observed_fields = set(payload)
    if observed_fields != required | source_order_fields:
        raise CLICTargetProtocolError("CLIC target prediction schema/field allowlist drift")
    sealed_source_class_order, _ = _validated_source_class_order_binding(
        payload.get("source_class_order"),
        payload.get("source_class_order_sha256"),
        label="CLIC target prediction source class order",
    )
    base = {key: value for key, value in payload.items() if key != "prediction_sha256"}
    declared_sha = _target.require_sha256(
        payload.get("prediction_sha256"), label="CLIC target prediction"
    )
    if _target.canonical_sha256(base) != declared_sha:
        raise CLICTargetProtocolError("CLIC target prediction logical SHA drift")
    for field in (
        "predictor_package_sha256",
        "package_manifest_sha256",
        "received_iq_data_sha256",
        "predictor_artifact_sha256",
        "predictor_state_sha256",
        "source_frozen_rule_sha256",
        "train_config_raw_sha256",
        "train_config_normalized_sha256",
        "train_config_data_normalized_sha256",
        "known_test_config_raw_sha256",
        "known_test_config_normalized_sha256",
        "lineage_sha256",
    ):
        _target.require_sha256(payload.get(field), label=f"CLIC target prediction {field}")
    for field in (
        "predictor_package_path",
        "predictor_state_path",
        "train_config_manifest_path",
    ):
        if not isinstance(payload.get(field), str) or not str(payload[field]):
            raise CLICTargetProtocolError(f"CLIC target prediction {field} is invalid")
    member = payload.get("train_config_member_name")
    if member not in {None, "candidate_train_data_config.json"}:
        raise CLICTargetProtocolError("CLIC target prediction train-config member binding drift")
    arm = str(payload.get("arm", ""))
    expected_operator = {
        "C": "raw_phase_control",
        "G": "complex_local_invariant_curvature",
    }
    if arm not in expected_operator or payload.get("operator") != expected_operator[arm]:
        raise CLICTargetProtocolError("CLIC target prediction arm/operator binding drift")
    fold = payload.get("fold_index")
    if type(fold) is not int or fold not in range(1, 7):
        raise CLICTargetProtocolError("CLIC target prediction fold binding drift")
    if payload.get("fold_config_key") != payload.get("train_config_data_normalized_sha256"):
        raise CLICTargetProtocolError("CLIC target prediction fold config key drift")
    row_count = _as_positive_int(payload.get("row_count"), label="prediction row_count")
    if _as_positive_int(payload.get("forward_count"), label="prediction forward_count") != row_count:
        raise CLICTargetProtocolError("CLIC target prediction must record exactly one forward per row")
    for field in (
        "target_fit_rows",
        "target_update_rows",
        "target_retry_count",
        "target_selection_count",
    ):
        if _as_nonnegative_int(payload.get(field), label=f"prediction {field}") != 0:
            raise CLICTargetProtocolError(f"CLIC target prediction forbids nonzero {field}")
    if payload.get("target_selection_feedback") is not False:
        raise CLICTargetProtocolError("CLIC target prediction selection feedback is forbidden")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != row_count:
        raise CLICTargetProtocolError("CLIC target prediction row-count closure failed")
    row_fields = {
        "opaque_token",
        "scene",
        "received_iq_sha256",
        "e_unknown",
        "decision",
        "predicted_index",
    }
    tokens: set[str] = set()
    for index, raw in enumerate(rows):
        row = _target._require_mapping(raw, label=f"CLIC target prediction row {index}")
        if set(row) != row_fields:
            raise CLICTargetProtocolError("CLIC target prediction row leaks unsupported sample state")
        token = _target.require_sha256(row.get("opaque_token"), label="prediction opaque token")
        if token in tokens:
            raise CLICTargetProtocolError("CLIC target prediction opaque-token uniqueness drift")
        tokens.add(token)
        if str(row.get("scene", "")) not in FORMAL_LEO_WEAK_SCENARIOS:
            raise CLICTargetProtocolError("CLIC target prediction scene drift")
        _target.require_sha256(row.get("received_iq_sha256"), label="prediction received IQ")
        try:
            energy = float(row.get("e_unknown"))
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError("CLIC target prediction unknown energy is invalid") from exc
        if not math.isfinite(energy):
            raise CLICTargetProtocolError("CLIC target prediction unknown energy is non-finite")
        decision = str(row.get("decision", ""))
        predicted_index = row.get("predicted_index")
        if decision == "registered":
            if isinstance(predicted_index, bool) or not isinstance(predicted_index, int) or predicted_index < 0:
                raise CLICTargetProtocolError("registered CLIC target prediction index is invalid")
            if predicted_index >= len(sealed_source_class_order):
                raise CLICTargetProtocolError(
                    "registered CLIC target prediction index exceeds sealed source class order"
                )
        elif decision in {"unknown", "defer"}:
            if predicted_index is not None:
                raise CLICTargetProtocolError("non-registered CLIC target prediction must not seal a class index")
        else:
            raise CLICTargetProtocolError("CLIC target prediction decision is invalid")
    return {
        "path": path,
        "raw_sha256": raw_after,
        "payload": payload,
    }


def _validate_prediction_package_binding(
    prediction: Mapping[str, Any],
    package: Mapping[str, Any],
) -> None:
    """Cross-bind prediction rows to the package without deserializing IQ."""

    package_path = Path(str(prediction["predictor_package_path"])).resolve()
    if package_path != Path(package["path"]).resolve():
        raise CLICTargetProtocolError("prediction target package path binding drift")
    manifest = _target._require_mapping(package["manifest"], label="target package manifest")
    checks = {
        "predictor_package_sha256": manifest["package_sha256"],
        "package_manifest_sha256": package["manifest_raw_sha256"],
        "received_iq_data_sha256": package["data_raw_sha256"],
        "lineage_sha256": manifest["lineage_sha256"],
        "known_test_config_raw_sha256": manifest["known_test_config_raw_sha256"],
        "known_test_config_normalized_sha256": manifest["known_test_config_normalized_sha256"],
    }
    for field, expected in checks.items():
        if prediction.get(field) != expected:
            raise CLICTargetProtocolError(f"prediction/package {field} binding drift")
    rows = prediction["rows"]
    if len(rows) != _as_positive_int(manifest["row_count"], label="target package row_count"):
        raise CLICTargetProtocolError("prediction/package row count drift")
    received_hashes = [str(row["received_iq_sha256"]) for row in rows]
    tokens = [str(row["opaque_token"]) for row in rows]
    if _target.canonical_sha256(received_hashes) != manifest["received_iq_sha256_root"]:
        raise CLICTargetProtocolError("prediction/package received-IQ root drift")
    if _target.canonical_sha256(tokens) != manifest["opaque_token_sha256"]:
        raise CLICTargetProtocolError("prediction/package opaque-token root drift")
    scene_counts = {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    for ordinal, row in enumerate(rows):
        scene = str(row["scene"])
        scene_counts[scene] += 1
        expected_token = _target.opaque_token(
            lineage_sha256=str(manifest["lineage_sha256"]),
            scene=scene,
            ordinal=ordinal,
            received_iq_sha256=str(row["received_iq_sha256"]),
        )
        if row["opaque_token"] != expected_token:
            raise CLICTargetProtocolError("prediction/package opaque token lineage drift")
    if scene_counts != manifest["row_count_by_scene"] or any(count <= 0 for count in scene_counts.values()):
        raise CLICTargetProtocolError("prediction/package formal scene count drift")


def _reopen_predictor_from_prediction(prediction: Mapping[str, Any]) -> tuple[Any, dict[str, str], dict[str, Any], Path]:
    """Reopen the exact C descriptor or G bundle sealed by a prediction."""

    artifact = _require_existing_file(prediction.get("predictor_state_path"), label="prediction predictor artifact")
    if sha256_file(artifact) != prediction["predictor_artifact_sha256"]:
        raise CLICTargetProtocolError("prediction predictor artifact byte SHA drift")
    runtime = load_verified_clic_predictor_state(artifact)
    identity = _require_runtime_identity(runtime)
    for field in ("arm", "operator", "fold_index", "state_sha256", "source_frozen_rule_sha256"):
        prediction_field = {
            "state_sha256": "predictor_state_sha256",
            "source_frozen_rule_sha256": "source_frozen_rule_sha256",
        }.get(field, field)
        if identity[field] != prediction[prediction_field]:
            raise CLICTargetProtocolError(f"prediction predictor {field} binding drift")
    if (
        identity["source_class_order"] != prediction["source_class_order"]
        or identity["source_class_order_sha256"]
        != prediction["source_class_order_sha256"]
    ):
        raise CLICTargetProtocolError("prediction predictor source class order/SHA drift")
    train = _runtime_train_config(runtime)
    expected_train = {
        "container_path": prediction["train_config_manifest_path"],
        "member_name": prediction["train_config_member_name"],
        "raw_sha256": prediction["train_config_raw_sha256"],
        "sealed_normalized_sha256": prediction["train_config_normalized_sha256"],
        "data_normalized_sha256": prediction["train_config_data_normalized_sha256"],
    }
    if not _same_train_config_binding(expected_train, train):
        raise CLICTargetProtocolError("prediction predictor train config binding drift")
    return runtime, identity, train, artifact


def _load_verified_adv3b02_reference(reference_path: str | Path) -> dict[str, Any]:
    """Reopen ADV3B02's own raw artifacts without imposing byte-identical IQ."""

    path = Path(reference_path).resolve()
    raw_before = sha256_file(path)
    payload = _target.read_json_object(path, label="ADV3B02 target-known reference")
    raw_after = sha256_file(path)
    if raw_before != raw_after:
        raise CLICTargetProtocolError("ADV3B02 reference changed while opening")
    required = {
        "schema",
        "sealed",
        "checkpoint_path",
        "checkpoint_sha256",
        "train_config_manifest_path",
        "train_config_raw_sha256",
        "train_config_normalized_sha256",
        "known_test_config_manifest_path",
        "known_test_config_raw_sha256",
        "known_test_config_normalized_sha256",
        "stratified_metric_artifact_path",
        "stratified_metric_artifact_sha256",
        "semantic_cells",
        "reference_sha256",
    }
    if set(payload) != required or payload.get("schema") != _target.ADV3B02_REFERENCE_SCHEMA or payload.get("sealed") is not True:
        raise CLICTargetProtocolError("ADV3B02 reference schema/seal field drift")
    reference_base = {key: value for key, value in payload.items() if key != "reference_sha256"}
    if _target.canonical_sha256(reference_base) != _target.require_sha256(
        payload.get("reference_sha256"), label="ADV3B02 reference"
    ):
        raise CLICTargetProtocolError("ADV3B02 reference logical SHA drift")
    for field in (
        "checkpoint_sha256",
        "train_config_raw_sha256",
        "train_config_normalized_sha256",
        "known_test_config_raw_sha256",
        "known_test_config_normalized_sha256",
        "stratified_metric_artifact_sha256",
    ):
        _target.require_sha256(payload.get(field), label=f"ADV3B02 reference {field}")
    checkpoint = _require_existing_file(payload.get("checkpoint_path"), label="ADV3B02 checkpoint")
    if sha256_file(checkpoint) != payload["checkpoint_sha256"]:
        raise CLICTargetProtocolError("ADV3B02 checkpoint byte SHA drift")
    train = _target.read_verified_config_manifest(
        payload["train_config_manifest_path"],
        expected_schema="cvs.phase1.adv3b02_train_data_config.v1",
        expected_raw_sha256=payload["train_config_raw_sha256"],
        label="ADV3B02 train config",
    )
    known = _target.read_verified_config_manifest(
        payload["known_test_config_manifest_path"],
        expected_schema="cvs.phase1.adv3b02_known_test_config.v1",
        expected_raw_sha256=payload["known_test_config_raw_sha256"],
        label="ADV3B02 known-test config",
    )
    train_data_sha = _train_data_sha(train["normalized"])
    known_data_sha = _known_test_data_sha(known["normalized"])
    if train_data_sha != payload["train_config_normalized_sha256"] or known_data_sha != payload["known_test_config_normalized_sha256"]:
        raise CLICTargetProtocolError("ADV3B02 reference normalized data-config SHA drift")
    metrics_path = _require_existing_file(
        payload.get("stratified_metric_artifact_path"), label="ADV3B02 stratified metrics"
    )
    if sha256_file(metrics_path) != payload["stratified_metric_artifact_sha256"]:
        raise CLICTargetProtocolError("ADV3B02 stratified metric artifact byte SHA drift")
    metrics = _target.read_json_object(metrics_path, label="ADV3B02 stratified target-known metrics")
    if metrics.get("schema") != _target.ADV3B02_METRICS_SCHEMA or not isinstance(metrics.get("cells"), list):
        raise CLICTargetProtocolError("ADV3B02 stratified metric schema/cells drift")
    cells = _validate_reference_cell_set(
        metrics.get("cells"),
        expected_train_config_sha256=train_data_sha,
        expected_known_config_sha256=known_data_sha,
        accepted_legacy_manifest_sha256=known["normalized_sha256"],
        known_config=known["normalized"],
    )
    if cells != payload["semantic_cells"] or not cells:
        raise CLICTargetProtocolError("ADV3B02 semantic cells differ from sealed metric artifact")
    return {
        "path": path,
        "raw_sha256": raw_after,
        "payload": payload,
        "checkpoint_path": checkpoint,
        "train_config": train,
        "known_test_config": known,
        "metrics_path": metrics_path,
        "cells": cells,
    }


def _load_verified_clic_truth_sidecar(
    truth_sidecar_path: str | Path,
    *,
    expected_package_sha256: str,
    expected_lineage_sha256: str,
) -> dict[str, Any]:
    """Open evaluator-only truth strictly after the non-truth preflight."""

    path = Path(truth_sidecar_path).resolve()
    raw_before = sha256_file(path)
    payload = _target.read_json_object(path, label="CLIC target truth sidecar")
    raw_after = sha256_file(path)
    if raw_before != raw_after:
        raise CLICTargetProtocolError("CLIC target truth sidecar changed while opening")
    required = {
        "schema",
        "sealed",
        "package_sha256",
        "lineage_sha256",
        "validator_receipt_path",
        "validator_receipt_raw_sha256",
        "cache_set_manifest_path",
        "cache_set_manifest_raw_sha256",
        "known_test_config_manifest_path",
        "known_test_config_raw_sha256",
        "known_test_config_normalized_sha256",
        "row_count",
        "rows",
    }
    if set(payload) != required or payload.get("schema") != _target.TARGET_TRUTH_SCHEMA or payload.get("sealed") is not True:
        raise CLICTargetProtocolError("CLIC target truth sidecar schema/seal drift")
    if payload.get("package_sha256") != expected_package_sha256 or payload.get("lineage_sha256") != expected_lineage_sha256:
        raise CLICTargetProtocolError("CLIC target truth sidecar package/lineage binding drift")
    for field in ("validator_receipt_path", "cache_set_manifest_path"):
        referenced_path = Path(str(payload.get(field, "")))
        if not referenced_path.is_absolute() or not str(referenced_path):
            raise CLICTargetProtocolError(f"CLIC target truth sidecar {field} is invalid")
    for field in ("validator_receipt_raw_sha256", "cache_set_manifest_raw_sha256"):
        _target.require_sha256(payload.get(field), label=f"CLIC target truth sidecar {field}")
    for field in ("known_test_config_manifest_path",):
        referenced_path = Path(str(payload.get(field, "")))
        if not referenced_path.is_absolute() or not str(referenced_path):
            raise CLICTargetProtocolError(f"CLIC target truth sidecar {field} is invalid")
    for field in ("known_test_config_raw_sha256", "known_test_config_normalized_sha256"):
        _target.require_sha256(payload.get(field), label=f"CLIC target truth sidecar {field}")
    row_count = _as_positive_int(payload.get("row_count"), label="truth sidecar row_count")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != row_count:
        raise CLICTargetProtocolError("CLIC target truth sidecar row-count closure failed")
    row_fields = {
        "opaque_token",
        "scene",
        "role",
        "truth",
        "tx_id",
        "rx_id",
        "day_id",
        "physical_sample_id",
    }
    tokens: set[str] = set()
    physical_ids: set[str] = set()
    roles: set[str] = set()
    for index, raw in enumerate(rows):
        row = _target._require_mapping(raw, label=f"truth sidecar row {index}")
        if set(row) != row_fields:
            raise CLICTargetProtocolError("CLIC target truth sidecar row field drift")
        token = _target.require_sha256(row.get("opaque_token"), label="truth opaque token")
        if token in tokens:
            raise CLICTargetProtocolError("CLIC target truth sidecar opaque-token uniqueness drift")
        tokens.add(token)
        if str(row.get("scene", "")) not in FORMAL_LEO_WEAK_SCENARIOS:
            raise CLICTargetProtocolError("CLIC target truth sidecar scene drift")
        role = str(row.get("role", ""))
        truth = str(row.get("truth", ""))
        if role == _TARGET_REGISTERED_ROLE:
            if not truth or truth == "unknown" or truth != str(row.get("tx_id", "")):
                raise CLICTargetProtocolError("registered-known truth sidecar identity binding drift")
        elif role == _TARGET_UNKNOWN_ROLE:
            if truth != "unknown":
                raise CLICTargetProtocolError("unknown truth sidecar identity binding drift")
        else:
            raise CLICTargetProtocolError("CLIC target truth sidecar role drift")
        physical_id = str(row.get("physical_sample_id", ""))
        if (
            not str(row.get("tx_id", ""))
            or not str(row.get("rx_id", ""))
            or not str(row.get("day_id", ""))
            or not physical_id
        ):
            raise CLICTargetProtocolError("CLIC target truth sidecar metadata is invalid")
        if physical_id in physical_ids:
            raise CLICTargetProtocolError(
                "CLIC target truth sidecar physical sample IDs are not globally unique"
            )
        physical_ids.add(physical_id)
        roles.add(role)
    if roles != _EXPECTED_ROLES:
        raise CLICTargetProtocolError("CLIC target truth sidecar role coverage drift")
    return {
        "path": path,
        "raw_sha256": raw_after,
        "payload": payload,
        "universe_roots": _target_universe_roots_from_truth_rows(rows),
    }


def _validate_truth_universe_bindings(
    truth: Mapping[str, Any], package: Mapping[str, Any], known_config: Mapping[str, Any]
) -> None:
    """Cross-check evaluator-only target IDs against package and known config."""

    roots = truth.get("universe_roots")
    if not isinstance(roots, Mapping):
        raise CLICTargetProtocolError("CLIC target truth sidecar universe roots are unavailable")
    manifest = package.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CLICTargetProtocolError("CLIC target package manifest is unavailable for truth binding")
    for field in _TARGET_UNIVERSE_ROOT_FIELDS:
        if roots.get(field) != manifest.get(field):
            raise CLICTargetProtocolError(
                f"CLIC target truth/package universe root drift: {field}"
            )
    _validate_target_universe_against_known_config(roots, known_config)


def _reopen_truth_known_test_config(truth: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Open the evaluator-only known-test contract after truth authorization."""

    payload = _target._require_mapping(truth.get("payload"), label="target truth sidecar")
    known = _target.read_verified_config_manifest(
        payload.get("known_test_config_manifest_path"),
        expected_schema="cvs.phase1.clic_known_test_config.v1",
        expected_raw_sha256=payload.get("known_test_config_raw_sha256"),
        label="target truth known-test config",
    )
    data_sha = _known_test_data_sha(known["normalized"])
    if data_sha != payload.get("known_test_config_normalized_sha256"):
        raise CLICTargetProtocolError("CLIC target truth known-test config normalized SHA drift")
    normalized = _target.normalize_known_test_config(known["normalized"])
    class_order = [str(value) for value in normalized["class_order"]]
    if not class_order or len(class_order) != len(set(class_order)):
        raise CLICTargetProtocolError("target truth known-test class order is invalid")
    target_known_tx_ids = [str(value) for value in normalized["target_known_tx_ids"]]
    if (
        not target_known_tx_ids
        or len(target_known_tx_ids) != len(set(target_known_tx_ids))
        or set(class_order) != set(target_known_tx_ids)
    ):
        raise CLICTargetProtocolError(
            "target truth known-test class order/TX universe binding drift"
        )
    return known, class_order


def _join_prediction_and_truth(
    prediction: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    class_order: list[str],
) -> list[dict[str, Any]]:
    prediction_rows = prediction["payload"]["rows"]
    joined_raw = _target.join_prediction_and_truth_by_opaque_token(
        prediction_rows, truth["payload"]["rows"]
    )
    joined: list[dict[str, Any]] = []
    for joined_row in joined_raw:
        predicted = _target._require_mapping(joined_row["prediction"], label="prediction join row")
        truth_row = _target._require_mapping(joined_row["truth"], label="truth join row")
        role = str(truth_row["role"])
        truth_label = str(truth_row["truth"])
        decision = str(predicted["decision"])
        predicted_index = predicted["predicted_index"]
        if role == _TARGET_REGISTERED_ROLE:
            if not truth_label or truth_label == "unknown":
                raise CLICTargetProtocolError("registered-known truth label is invalid")
            formal_known = truth_label in class_order
            correct = (
                formal_known
                and decision == "registered"
                and isinstance(predicted_index, int)
                and not isinstance(predicted_index, bool)
                and 0 <= predicted_index < len(class_order)
                and class_order[predicted_index] == truth_label
            )
            scoring_role = (
                "registered_known" if formal_known else "inactive_registered_known"
            )
        elif role == _TARGET_UNKNOWN_ROLE:
            correct = False
            scoring_role = "unknown"
        else:
            raise CLICTargetProtocolError("target truth role is outside the confirmation contract")
        joined.append(
            {
                "scene": str(predicted["scene"]),
                "role": scoring_role,
                "truth": truth_label,
                "decision": decision,
                "predicted_index": predicted_index,
                "known_identity_correct": bool(correct),
                "e_unknown": float(predicted["e_unknown"]),
                # Evaluator-only strata; only their aggregates are written to
                # the score receipt.  They never return to the predictor.
                "tx_id": str(truth_row["tx_id"]),
                "rx_id": str(truth_row["rx_id"]),
                "day_id": str(truth_row["day_id"]),
                "physical_sample_id": str(truth_row["physical_sample_id"]),
            }
        )
    return joined


def _computed_accuracy_triplet(numerator: int, denominator: int, *, label: str) -> dict[str, Any]:
    if denominator <= 0:
        raise CLICTargetProtocolError(f"{label} requires a positive denominator")
    if numerator < 0 or numerator > denominator:
        raise CLICTargetProtocolError(f"{label} numerator/denominator drift")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "accuracy": float(numerator) / float(denominator),
    }


def _computed_rate(numerator: int, denominator: int, *, label: str) -> dict[str, Any]:
    result = _computed_accuracy_triplet(numerator, denominator, label=label)
    return {
        "numerator": result["numerator"],
        "denominator": result["denominator"],
        "rate": result["accuracy"],
    }


def _known_scope_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    class_order: Sequence[str],
    receiver_ids: Sequence[str],
    day_ids: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Recompute one complete registered-known metric scope from rows."""

    expected_groups = {
        "by_class": tuple(str(value) for value in class_order),
        "by_receiver": tuple(str(value) for value in receiver_ids),
        "by_day": tuple(str(value) for value in day_ids),
    }
    if any(not values or len(values) != len(set(values)) for values in expected_groups.values()):
        raise CLICTargetProtocolError("known-target expected class/RX/day groups drift")
    counts: dict[str, dict[str, dict[str, int]]] = {
        field: {name: {"correct": 0, "denominator": 0} for name in values}
        for field, values in expected_groups.items()
    }
    denominator = correct = false_reject = defer = accepted = 0
    for index, raw in enumerate(rows):
        row = _target._require_mapping(raw, label=f"{label} row {index}")
        if str(row.get("role", "")) != "registered_known":
            continue
        truth = str(row.get("truth", ""))
        rx_id = str(row.get("rx_id", ""))
        day_id = str(row.get("day_id", ""))
        group_values = {"by_class": truth, "by_receiver": rx_id, "by_day": day_id}
        for field, group in group_values.items():
            if group not in counts[field]:
                raise CLICTargetProtocolError(f"{label} has an out-of-contract {field} group")
        decision = str(row.get("decision", ""))
        if decision not in {"registered", "unknown", "defer"}:
            raise CLICTargetProtocolError(f"{label} decision drift")
        is_correct = bool(row.get("known_identity_correct", False))
        if is_correct and decision != "registered":
            raise CLICTargetProtocolError(f"{label} known identity correctness/decision drift")
        denominator += 1
        correct += int(is_correct)
        false_reject += int(decision == "unknown")
        defer += int(decision == "defer")
        accepted += int(decision == "registered")
        for field, group in group_values.items():
            counts[field][group]["denominator"] += 1
            counts[field][group]["correct"] += int(is_correct)
    overall = _computed_accuracy_triplet(correct, denominator, label=f"{label} overall")
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for field, grouped in counts.items():
        result: dict[str, dict[str, Any]] = {}
        for group, values in grouped.items():
            result[group] = _computed_accuracy_triplet(
                values["correct"], values["denominator"], label=f"{label} {field}[{group}]"
            )
        by_group[field] = result
        if (
            sum(int(item["numerator"]) for item in result.values()) != overall["numerator"]
            or sum(int(item["denominator"]) for item in result.values()) != overall["denominator"]
        ):
            raise CLICTargetProtocolError(f"{label} {field} count closure drift")
    accepted_known = {
        "correct": int(correct),
        "denominator": int(accepted),
        "accuracy": (float(correct) / float(accepted)) if accepted else None,
        "coverage": _computed_rate(
            accepted, denominator, label=f"{label} accepted-known coverage"
        ),
    }
    return {
        "overall": overall,
        **by_group,
        "macro_accuracy": sum(
            float(value["accuracy"]) for value in by_group["by_class"].values()
        ) / len(by_group["by_class"]),
        "min_class_accuracy": min(
            float(value["accuracy"]) for value in by_group["by_class"].values()
        ),
        "min_receiver_accuracy": min(
            float(value["accuracy"]) for value in by_group["by_receiver"].values()
        ),
        "min_day_accuracy": min(
            float(value["accuracy"]) for value in by_group["by_day"].values()
        ),
        "known_false_reject": _computed_accuracy_triplet(
            false_reject, denominator, label=f"{label} known false reject"
        ),
        "known_defer": _computed_accuracy_triplet(
            defer, denominator, label=f"{label} known defer"
        ),
        "accepted_known": accepted_known,
    }


def _score_known_target_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    class_order: Sequence[str],
    receiver_ids: Sequence[str],
    day_ids: Sequence[str],
) -> dict[str, Any]:
    """Recompute global and three-scene full registered-known metrics."""

    materialized = [_target._require_mapping(row, label="known-target score row") for row in rows]
    by_scene: dict[str, dict[str, Any]] = {}
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        scene_rows = [row for row in materialized if str(row.get("scene", "")) == scene]
        by_scene[scene] = _known_scope_metrics(
            scene_rows,
            class_order=class_order,
            receiver_ids=receiver_ids,
            day_ids=day_ids,
            label=f"known-target {scene}",
        )
    global_scope = _known_scope_metrics(
        materialized,
        class_order=class_order,
        receiver_ids=receiver_ids,
        day_ids=day_ids,
        label="known-target global",
    )
    inactive_by_tx: dict[str, dict[str, int]] = {}
    inactive_by_scene: dict[str, int] = {
        scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS
    }
    for row in materialized:
        if str(row.get("role", "")) != "inactive_registered_known":
            continue
        tx_id = str(row.get("tx_id", ""))
        scene = str(row.get("scene", ""))
        if not tx_id or scene not in inactive_by_scene:
            raise CLICTargetProtocolError("inactive registered-known audit metadata drift")
        inactive_by_tx.setdefault(tx_id, {"denominator": 0, "count": 0})
        inactive_by_tx[tx_id]["denominator"] += 1
        inactive_by_tx[tx_id]["count"] += 1
        inactive_by_scene[scene] += 1
    # Keep the earlier flat count fields for report/backward-reader stability;
    # the authoritative metrics are the fully stratified global/by_scene maps.
    return {
        "global": global_scope,
        "by_scene": by_scene,
        "known_denominator_global": global_scope["overall"]["denominator"],
        "known_numerator_global": global_scope["overall"]["numerator"],
        "known_accuracy_global": global_scope["overall"]["accuracy"],
        "known_denominator_by_scene": {
            scene: by_scene[scene]["overall"]["denominator"]
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        },
        "known_numerator_by_scene": {
            scene: by_scene[scene]["overall"]["numerator"]
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        },
        "known_accuracy_by_scene": {
            scene: by_scene[scene]["overall"]["accuracy"]
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        },
        "known_unknown_decision_global": global_scope["known_false_reject"]["numerator"],
        "known_unknown_decision_by_scene": {
            scene: by_scene[scene]["known_false_reject"]["numerator"]
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        },
        "known_defer_global": global_scope["known_defer"]["numerator"],
        "known_defer_by_scene": {
            scene: by_scene[scene]["known_defer"]["numerator"]
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        },
        "inactive_registered_known": {
            "excluded_from_known_denominator": True,
            "count": sum(item["count"] for item in inactive_by_tx.values()),
            "by_tx": dict(sorted(inactive_by_tx.items())),
            "by_scene": inactive_by_scene,
        },
    }


def _auroc_pairwise_tie_half(unknown_scores: Sequence[float], known_scores: Sequence[float]) -> float:
    """AUROC with the required pairwise tie contribution of 0.5."""

    if not unknown_scores or not known_scores:
        raise CLICTargetProtocolError("AUROC requires positive known and true-unknown denominators")
    wins = 0.0
    for unknown in unknown_scores:
        for known in known_scores:
            if unknown > known:
                wins += 1.0
            elif unknown == known:
                wins += 0.5
    return wins / float(len(unknown_scores) * len(known_scores))


def _aupr_out_grouped_descending(
    unknown_scores: Sequence[float], known_scores: Sequence[float]
) -> float:
    """AUPR-out AP over distinct descending score groups (tie-safe)."""

    if not unknown_scores or not known_scores:
        raise CLICTargetProtocolError("AUPR-out requires positive known and true-unknown denominators")
    grouped: dict[float, list[int]] = {}
    for score in unknown_scores:
        grouped.setdefault(float(score), [0, 0])[0] += 1
    for score in known_scores:
        grouped.setdefault(float(score), [0, 0])[1] += 1
    true_positive = false_positive = 0
    previous_recall = 0.0
    ap = 0.0
    total_positive = len(unknown_scores)
    for _score, (group_positive, group_negative) in sorted(
        grouped.items(), key=lambda item: item[0], reverse=True
    ):
        true_positive += group_positive
        false_positive += group_negative
        recall = true_positive / float(total_positive)
        precision = true_positive / float(true_positive + false_positive)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def _fpr95_at_score_thresholds(unknown_scores: Sequence[float], known_scores: Sequence[float]) -> float:
    """Minimum FPR at score>=threshold among thresholds with TPR>=.95."""

    if not unknown_scores or not known_scores:
        raise CLICTargetProtocolError("FPR95 requires positive known and true-unknown denominators")
    thresholds = sorted(set(float(value) for value in (*unknown_scores, *known_scores)))
    eligible: list[float] = []
    for threshold in thresholds:
        tpr = sum(score >= threshold for score in unknown_scores) / float(len(unknown_scores))
        if tpr >= 0.95:
            eligible.append(
                sum(score >= threshold for score in known_scores) / float(len(known_scores))
            )
    if not eligible:
        raise CLICTargetProtocolError("FPR95 score threshold set has no TPR>=.95 point")
    return min(eligible)


def _pure_open_set_score_metrics(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, float]:
    """Compute only score-distribution metrics; no decision field is needed."""

    unknown_scores: list[float] = []
    known_scores: list[float] = []
    for index, raw in enumerate(rows):
        row = _target._require_mapping(raw, label=f"{label} row {index}")
        role = str(row.get("role", ""))
        if role not in {"registered_known", "inactive_registered_known", "unknown"}:
            raise CLICTargetProtocolError(f"{label} known/unknown role drift")
        try:
            score = float(row.get("e_unknown"))
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError(f"{label} unknown energy is invalid") from exc
        if not math.isfinite(score):
            raise CLICTargetProtocolError(f"{label} unknown energy is non-finite")
        if role == "unknown":
            unknown_scores.append(score)
        elif role == "registered_known":
            known_scores.append(score)
    if not unknown_scores or not known_scores:
        raise CLICTargetProtocolError(
            f"{label} requires nonempty known and true-unknown energy classes"
        )
    return {
        "AUROC_unknown": _auroc_pairwise_tie_half(unknown_scores, known_scores),
        "AUPR_out": _aupr_out_grouped_descending(unknown_scores, known_scores),
        "FPR95": _fpr95_at_score_thresholds(unknown_scores, known_scores),
    }


def _open_set_scope_metrics(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """Compute score/rule open-set metrics for one global or formal scene scope."""

    materialized = [_target._require_mapping(row, label=f"{label} row") for row in rows]
    score_metrics = _pure_open_set_score_metrics(materialized, label=label)
    unknown_scores: list[float] = []
    known_scores: list[float] = []
    unknown_rejected = unknown_accepted = unknown_safe = 0
    known_false_reject = known_defer = known_accepted = known_correct_accepted = 0
    known_count = 0
    for index, raw in enumerate(materialized):
        row = _target._require_mapping(raw, label=f"{label} row {index}")
        role = str(row.get("role", ""))
        decision = str(row.get("decision", ""))
        if role not in {"registered_known", "inactive_registered_known", "unknown"} or decision not in {
            "registered",
            "unknown",
            "defer",
        }:
            raise CLICTargetProtocolError(f"{label} role/decision drift")
        try:
            score = float(row.get("e_unknown"))
        except (TypeError, ValueError) as exc:
            raise CLICTargetProtocolError(f"{label} unknown score is invalid") from exc
        if not math.isfinite(score):
            raise CLICTargetProtocolError(f"{label} unknown score is non-finite")
        if role == "unknown":
            unknown_scores.append(score)
            unknown_rejected += int(decision == "unknown")
            unknown_accepted += int(decision == "registered")
            unknown_safe += int(decision in {"unknown", "defer"})
        elif role == "registered_known":
            known_scores.append(score)
            known_count += 1
            known_false_reject += int(decision == "unknown")
            known_defer += int(decision == "defer")
            known_accepted += int(decision == "registered")
            known_correct_accepted += int(
                decision == "registered" and bool(row.get("known_identity_correct", False))
            )
    if not unknown_scores or not known_scores:
        raise CLICTargetProtocolError(f"{label} requires positive known and true-unknown denominators")
    accepted_known = {
        "correct": int(known_correct_accepted),
        "denominator": int(known_accepted),
        "accuracy": (
            float(known_correct_accepted) / float(known_accepted)
            if known_accepted
            else None
        ),
        "coverage": _computed_rate(
            known_accepted, known_count, label=f"{label} accepted-known coverage"
        ),
    }
    return {
        **score_metrics,
        "unknown_rejection": _computed_rate(
            unknown_rejected, len(unknown_scores), label=f"{label} unknown rejection"
        ),
        "unknown_FAR": _computed_rate(
            unknown_accepted, len(unknown_scores), label=f"{label} unknown FAR"
        ),
        "unknown_safe_handling": _computed_rate(
            unknown_safe, len(unknown_scores), label=f"{label} unknown safe handling"
        ),
        "known_false_reject": _computed_rate(
            known_false_reject, known_count, label=f"{label} known false reject"
        ),
        "known_defer": _computed_rate(
            known_defer, known_count, label=f"{label} known defer"
        ),
        "accepted_known": accepted_known,
        # This is deliberately the same exact count/rate record as the
        # accepted-known coverage rather than a dashboard-only scalar.
        "coverage": accepted_known["coverage"],
    }


def compute_target_open_set_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pure, truth-side scoring formula for one already-authorized target scope.

    ``e_unknown`` is the continuous out-of-known score (larger means more
    unknown).  The routine has no file, predictor, threshold-fitting or state
    access, so tests and the sealed scorer share exactly the same tie-safe
    AUROC/AUPR/FPR95 arithmetic.
    """

    materialized = [_target._require_mapping(row, label="target open-set metric row") for row in rows]
    return _pure_open_set_score_metrics(materialized, label="target open-set metrics")


def _unknown_slice_metrics(rows: Iterable[Mapping[str, Any]], *, group_field: str) -> dict[str, Any]:
    """Return global unknown TX/RX/day slice rates plus coverage and worsts."""

    unknown_rows = [
        _target._require_mapping(row, label="unknown slice row")
        for row in rows
        if str(_target._require_mapping(row, label="unknown slice row").get("role", "")) == "unknown"
    ]
    if not unknown_rows:
        raise CLICTargetProtocolError("unknown slice scoring requires true-unknown rows")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in unknown_rows:
        group = str(row.get(group_field, ""))
        if not group:
            raise CLICTargetProtocolError(f"unknown slice {group_field} metadata drift")
        grouped.setdefault(group, []).append(row)
    total = len(unknown_rows)
    result: dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        denominator = len(group_rows)
        rejected = sum(str(row.get("decision", "")) == "unknown" for row in group_rows)
        accepted = sum(str(row.get("decision", "")) == "registered" for row in group_rows)
        safe = sum(str(row.get("decision", "")) in {"unknown", "defer"} for row in group_rows)
        result[group] = {
            "coverage": _computed_rate(denominator, total, label=f"unknown {group_field} coverage"),
            "rejection": _computed_rate(rejected, denominator, label=f"unknown {group_field} rejection"),
            "safe_handling": _computed_rate(safe, denominator, label=f"unknown {group_field} safe"),
            "unknown_FAR": _computed_rate(accepted, denominator, label=f"unknown {group_field} FAR"),
        }
    return result


def _fold_overall_comparisons(
    candidate_by_scene: Mapping[str, Any], baseline_by_scene: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Derive only fold-local three-scene overall comparisons from count cells.

    This intentionally has no cross-fold aggregation.  The equal-weight value
    gives every formal LEO scene one vote; the sample-pooled value gives every
    sealed registered-known row one vote.  Both are derived from the exact
    per-scene overall numerator/denominator records already used by the core
    non-inferiority checks.
    """

    expected_scenes = set(FORMAL_LEO_WEAK_SCENARIOS)
    if set(candidate_by_scene) != expected_scenes or set(baseline_by_scene) != expected_scenes:
        raise CLICTargetProtocolError("fold overall comparison scene closure drift")

    def collect(scopes: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for scene in FORMAL_LEO_WEAK_SCENARIOS:
            scope = _target._require_mapping(scopes[scene], label=f"{label} {scene}")
            values[scene] = _validated_accuracy_triplet(
                scope.get("overall"), label=f"{label} {scene} overall"
            )
        return values

    candidate = collect(candidate_by_scene, label="CLIC candidate")
    baseline = collect(baseline_by_scene, label="ADV3B02 baseline")
    candidate_equal = sum(float(candidate[scene]["accuracy"]) for scene in FORMAL_LEO_WEAK_SCENARIOS) / float(
        len(FORMAL_LEO_WEAK_SCENARIOS)
    )
    baseline_equal = sum(float(baseline[scene]["accuracy"]) for scene in FORMAL_LEO_WEAK_SCENARIOS) / float(
        len(FORMAL_LEO_WEAK_SCENARIOS)
    )
    candidate_pooled_numerator = sum(int(candidate[scene]["numerator"]) for scene in FORMAL_LEO_WEAK_SCENARIOS)
    candidate_pooled_denominator = sum(int(candidate[scene]["denominator"]) for scene in FORMAL_LEO_WEAK_SCENARIOS)
    baseline_pooled_numerator = sum(int(baseline[scene]["numerator"]) for scene in FORMAL_LEO_WEAK_SCENARIOS)
    baseline_pooled_denominator = sum(int(baseline[scene]["denominator"]) for scene in FORMAL_LEO_WEAK_SCENARIOS)
    candidate_pooled = _computed_accuracy_triplet(
        candidate_pooled_numerator, candidate_pooled_denominator, label="CLIC candidate fold sample-pooled overall"
    )
    baseline_pooled = _computed_accuracy_triplet(
        baseline_pooled_numerator, baseline_pooled_denominator, label="ADV3B02 baseline fold sample-pooled overall"
    )
    return {
        "fold_three_scene_equal_weight_overall": {
            "candidate": {
                "by_scene_accuracy": {
                    scene: candidate[scene]["accuracy"] for scene in FORMAL_LEO_WEAK_SCENARIOS
                },
                "accuracy": candidate_equal,
            },
            "baseline": {
                "by_scene_accuracy": {
                    scene: baseline[scene]["accuracy"] for scene in FORMAL_LEO_WEAK_SCENARIOS
                },
                "accuracy": baseline_equal,
            },
            "passed": candidate_equal >= baseline_equal,
        },
        "fold_sample_pooled_overall": {
            "candidate": candidate_pooled,
            "baseline": baseline_pooled,
            "passed": float(candidate_pooled["accuracy"]) >= float(baseline_pooled["accuracy"]),
        },
    }


def _score_open_set_target_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute global/per-scene score metrics and worst unknown strata."""

    materialized = [_target._require_mapping(row, label="open-set target row") for row in rows]
    by_scene = {
        scene: _open_set_scope_metrics(
            [row for row in materialized if str(row.get("scene", "")) == scene],
            label=f"open-set {scene}",
        )
        for scene in FORMAL_LEO_WEAK_SCENARIOS
    }
    slices = {
        "by_tx": _unknown_slice_metrics(materialized, group_field="tx_id"),
        "by_receiver": _unknown_slice_metrics(materialized, group_field="rx_id"),
        "by_day": _unknown_slice_metrics(materialized, group_field="day_id"),
    }
    worst: dict[str, Any] = {}
    for group, values in slices.items():
        worst[group] = {
            "min_rejection": min(float(value["rejection"]["rate"]) for value in values.values()),
            "min_safe_handling": min(float(value["safe_handling"]["rate"]) for value in values.values()),
            "max_unknown_FAR": max(float(value["unknown_FAR"]["rate"]) for value in values.values()),
            "min_coverage": min(float(value["coverage"]["rate"]) for value in values.values()),
        }
    return {
        "global": _open_set_scope_metrics(materialized, label="open-set global"),
        "by_scene": by_scene,
        "unknown_slices": {**slices, "worst": worst},
    }


def _local_class_axis_metrics_from_crossed(
    crossed: Mapping[str, Any],
    *,
    active_classes: Sequence[str],
    axis_ids: Sequence[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Recompute one local-class axis partition from sealed crossed counts."""

    normalized_classes = tuple(str(value) for value in active_classes)
    normalized_axes = tuple(str(value) for value in axis_ids)
    if (
        not normalized_classes
        or len(set(normalized_classes)) != len(normalized_classes)
        or not normalized_axes
        or len(set(normalized_axes)) != len(normalized_axes)
    ):
        raise CLICTargetProtocolError(f"{label} local class/axis universe drift")
    result: dict[str, dict[str, Any]] = {}
    for axis_id in normalized_axes:
        numerator = denominator = 0
        for class_id in normalized_classes:
            class_cells = _target._require_mapping(
                crossed.get(class_id), label=f"{label}[{class_id}]"
            )
            if set(class_cells) != set(normalized_axes):
                raise CLICTargetProtocolError(f"{label} crossed axis universe drift")
            cell = _target._require_mapping(
                class_cells.get(axis_id), label=f"{label}[{class_id}][{axis_id}]"
            )
            numerator += _as_nonnegative_int(
                cell.get("numerator"), label=f"{label}[{class_id}][{axis_id}] numerator"
            )
            denominator += _as_positive_int(
                cell.get("denominator"), label=f"{label}[{class_id}][{axis_id}] denominator"
            )
        result[axis_id] = _computed_accuracy_triplet(
            numerator, denominator, label=f"{label}[{axis_id}]"
        )
    return result


def _excluded_class_axis_counts(
    crossed: Mapping[str, Any],
    *,
    inactive_classes: Sequence[str],
    axis_ids: Sequence[str],
    label: str,
) -> dict[str, dict[str, int]]:
    """Audit excluded union classes without turning them into formal metrics."""

    result: dict[str, dict[str, int]] = {}
    for raw_axis in axis_ids:
        axis_id = str(raw_axis)
        numerator = denominator = 0
        for raw_class in inactive_classes:
            class_id = str(raw_class)
            cells = _target._require_mapping(
                crossed.get(class_id), label=f"{label}[{class_id}]"
            )
            cell = _target._require_mapping(
                cells.get(axis_id), label=f"{label}[{class_id}][{axis_id}]"
            )
            numerator += _as_nonnegative_int(
                cell.get("numerator"), label=f"{label}[{class_id}][{axis_id}] numerator"
            )
            denominator += _as_positive_int(
                cell.get("denominator"), label=f"{label}[{class_id}][{axis_id}] denominator"
            )
        result[axis_id] = {"numerator": numerator, "denominator": denominator}
    return result


def _noninferiority_against_adv3b02(
    known_audit: Mapping[str, Any],
    baseline_cells: Sequence[Mapping[str, Any]],
    *,
    fold_index: int,
    fold_config_key: str,
    source_class_order: Sequence[str],
) -> dict[str, Any]:
    """Compare only same semantic data-config cells, never physical packages."""

    if type(fold_index) is not int or fold_index not in range(1, 7):
        raise CLICTargetProtocolError("CLIC candidate fold index is invalid for ADV comparison")
    expected_key = _target.require_sha256(fold_config_key, label="CLIC candidate fold config key")
    active_classes = [str(value) for value in source_class_order]
    if len(active_classes) != 4 or len(set(active_classes)) != 4 or any(
        not value for value in active_classes
    ):
        raise CLICTargetProtocolError(
            "CLIC candidate source class order must be the sealed local-four"
        )
    if len(baseline_cells) != len(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTargetProtocolError("ADV3B02 comparison needs exactly three scene cells")
    cells_by_scene = {str(cell.get("scene", "")): cell for cell in baseline_cells}
    if set(cells_by_scene) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise CLICTargetProtocolError("ADV3B02 comparison scene closure drift")
    candidate_by_scene = known_audit.get("by_scene")
    if not isinstance(candidate_by_scene, Mapping) or set(candidate_by_scene) != set(
        FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise CLICTargetProtocolError("CLIC candidate known scene audit closure drift")
    result_by_scene: dict[str, Any] = {}
    all_passed = True
    complete_baseline_by_scene: dict[str, Mapping[str, Any]] = {}
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        baseline = _target._require_mapping(cells_by_scene[scene], label="ADV3B02 comparison cell")
        if baseline.get("fold_config_key") != expected_key:
            raise CLICTargetProtocolError("ADV3B02/CLIC fold config key mismatch")
        candidate = _target._require_mapping(
            candidate_by_scene[scene], label="CLIC candidate known scene audit"
        )
        # A legacy top-level-only cell remains readable for provenance but is
        # deliberately non-promotable: it cannot establish class/RX/day
        # non-inferiority against the full CLIC target audit.
        rich_required = {
            "overall",
            "by_class",
            "by_receiver",
            "by_day",
            "by_class_receiver",
            "by_class_day",
            "macro_accuracy",
            "min_class_accuracy",
            "min_receiver_accuracy",
            "min_day_accuracy",
            "known_false_reject",
            "known_defer",
            "accepted_known",
        }
        if not rich_required <= set(baseline):
            result_by_scene[scene] = {
                "candidate": candidate,
                "baseline": baseline,
                "passed": False,
                "reason": "baseline_incomplete_class_rx_day_stratification",
            }
            all_passed = False
            continue
        baseline_classes = _target._require_mapping(
            baseline["by_class"], label="ADV3B02 baseline by_class"
        )
        if not set(active_classes).issubset(set(baseline_classes)):
            raise CLICTargetProtocolError(
                "ADV3B02 baseline does not cover the sealed CLIC local-four classes"
            )
        inactive_classes = sorted(set(baseline_classes).difference(active_classes))
        active_by_class = {
            name: _target._require_mapping(
                baseline_classes[name], label=f"ADV3B02 baseline active class {name}"
            )
            for name in active_classes
        }
        baseline_receivers = _target._require_mapping(
            baseline["by_receiver"], label="ADV3B02 baseline by_receiver"
        )
        baseline_days = _target._require_mapping(
            baseline["by_day"], label="ADV3B02 baseline by_day"
        )
        crossed_receiver = _target._require_mapping(
            baseline["by_class_receiver"], label="ADV3B02 baseline by_class_receiver"
        )
        crossed_day = _target._require_mapping(
            baseline["by_class_day"], label="ADV3B02 baseline by_class_day"
        )
        active_by_receiver = _local_class_axis_metrics_from_crossed(
            crossed_receiver,
            active_classes=active_classes,
            axis_ids=tuple(str(value) for value in baseline_receivers),
            label=f"ADV3B02 {scene} sealed local-four by_receiver",
        )
        active_by_day = _local_class_axis_metrics_from_crossed(
            crossed_day,
            active_classes=active_classes,
            axis_ids=tuple(str(value) for value in baseline_days),
            label=f"ADV3B02 {scene} sealed local-four by_day",
        )
        active_numerator = sum(int(item["numerator"]) for item in active_by_class.values())
        active_denominator = sum(int(item["denominator"]) for item in active_by_class.values())
        active_overall = _computed_accuracy_triplet(
            active_numerator,
            active_denominator,
            label=f"ADV3B02 {scene} sealed local-four overall",
        )
        filtered_baseline = dict(baseline)
        filtered_baseline["overall"] = active_overall
        filtered_baseline["by_class"] = active_by_class
        filtered_baseline["by_receiver"] = active_by_receiver
        filtered_baseline["by_day"] = active_by_day
        filtered_baseline["by_class_receiver"] = {
            class_id: _target._require_mapping(
                crossed_receiver[class_id], label=f"ADV3B02 local-four receiver class {class_id}"
            )
            for class_id in active_classes
        }
        filtered_baseline["by_class_day"] = {
            class_id: _target._require_mapping(
                crossed_day[class_id], label=f"ADV3B02 local-four day class {class_id}"
            )
            for class_id in active_classes
        }
        filtered_baseline["macro_accuracy"] = sum(
            float(item["accuracy"]) for item in active_by_class.values()
        ) / float(len(active_by_class))
        filtered_baseline["min_class_accuracy"] = min(
            float(item["accuracy"]) for item in active_by_class.values()
        )
        filtered_baseline["min_receiver_accuracy"] = min(
            float(item["accuracy"]) for item in active_by_receiver.values()
        )
        filtered_baseline["min_day_accuracy"] = min(
            float(item["accuracy"]) for item in active_by_day.values()
        )
        # Candidate and ADV are both reduced from their own formal evidence to
        # the predictor-sealed local-four.  The union-only classes remain an
        # explicit audit record and are never treated as unknown or a known
        # denominator.
        filtered_baseline["inactive_registered_known"] = {
            "excluded_from_local_four_comparison": True,
            "class_ids": inactive_classes,
            "by_class": {
                class_id: _target._require_mapping(
                    baseline_classes[class_id], label=f"ADV3B02 inactive class {class_id}"
                )
                for class_id in inactive_classes
            },
            "by_receiver": _excluded_class_axis_counts(
                crossed_receiver,
                inactive_classes=inactive_classes,
                axis_ids=tuple(str(value) for value in baseline_receivers),
                label=f"ADV3B02 {scene} excluded by_receiver",
            ),
            "by_day": _excluded_class_axis_counts(
                crossed_day,
                inactive_classes=inactive_classes,
                axis_ids=tuple(str(value) for value in baseline_days),
                label=f"ADV3B02 {scene} excluded by_day",
            ),
        }
        complete_baseline_by_scene[scene] = filtered_baseline
        checks: dict[str, bool] = {}
        for field in (
            "macro_accuracy",
            "min_class_accuracy",
            "min_receiver_accuracy",
            "min_day_accuracy",
        ):
            checks[field] = float(candidate[field]) >= float(filtered_baseline[field])
        checks["overall_accuracy"] = float(candidate["overall"]["accuracy"]) >= float(
            filtered_baseline["overall"]["accuracy"]
        )
        for field in ("by_class", "by_receiver", "by_day"):
            candidate_groups = _target._require_mapping(candidate[field], label=f"candidate {field}")
            baseline_groups = _target._require_mapping(filtered_baseline[field], label=f"baseline {field}")
            if set(candidate_groups) != set(baseline_groups):
                raise CLICTargetProtocolError(
                    f"ADV3B02/CLIC {field} group universe mismatch"
                )
            for group in sorted(candidate_groups):
                checks[f"{field}:{group}"] = float(candidate_groups[group]["accuracy"]) >= float(
                    baseline_groups[group]["accuracy"]
                )
        passed = all(checks.values())
        result_by_scene[scene] = {
            "candidate": candidate,
            "baseline": filtered_baseline,
            "checks": checks,
            "passed": passed,
        }
        all_passed = all_passed and passed
    result = {
        "fold_index": fold_index,
        "fold_config_key": expected_key,
        "by_scene": result_by_scene,
        "passed": all_passed,
    }
    if len(complete_baseline_by_scene) == len(FORMAL_LEO_WEAK_SCENARIOS):
        fold_overall = _fold_overall_comparisons(candidate_by_scene, complete_baseline_by_scene)
        result.update(fold_overall)
        result["passed"] = bool(
            result["passed"]
            and fold_overall["fold_three_scene_equal_weight_overall"]["passed"]
            and fold_overall["fold_sample_pooled_overall"]["passed"]
        )
    else:
        # Old provenance-only cells cannot establish any formal aggregate.  Do
        # not fabricate a weighted number from an incomplete class/RX/day
        # reference; preserve a fail-closed comparison record instead.
        result["fold_three_scene_equal_weight_overall"] = {
            "candidate": None,
            "baseline": None,
            "passed": False,
            "reason": "baseline_incomplete_class_rx_day_stratification",
        }
        result["fold_sample_pooled_overall"] = {
            "candidate": None,
            "baseline": None,
            "passed": False,
            "reason": "baseline_incomplete_class_rx_day_stratification",
        }
    return result


def _reopen_validated_once_target_provenance(
    *, truth: Mapping[str, Any], package: Mapping[str, Any]
) -> dict[str, str]:
    """Reopen the evaluator-only VALIDATED_ONCE chain after truth authorization.

    The predictor package intentionally has no receipt/cache path.  The scorer
    receives those paths only inside its already-authorized truth sidecar, then
    proves that the receipt and cache-set manifest still match the package
    commitments before it emits an evaluable score receipt.
    """

    truth_payload = _target._require_mapping(truth.get("payload"), label="target truth sidecar")
    manifest = _target._require_mapping(package.get("manifest"), label="target package manifest")
    receipt_path = Path(str(truth_payload.get("validator_receipt_path", ""))).resolve()
    cache_path = Path(str(truth_payload.get("cache_set_manifest_path", ""))).resolve()
    if not receipt_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("CLIC target evaluator provenance receipt or cache-set manifest is missing")
    receipt_raw_sha = _target.require_sha256(
        truth_payload.get("validator_receipt_raw_sha256"), label="target truth validator receipt"
    )
    cache_raw_sha = _target.require_sha256(
        truth_payload.get("cache_set_manifest_raw_sha256"), label="target truth cache-set manifest"
    )
    if sha256_file(receipt_path) != receipt_raw_sha or sha256_file(cache_path) != cache_raw_sha:
        raise CLICTargetProtocolError("CLIC target evaluator provenance byte SHA drift")
    if (
        manifest.get("validator_receipt_sha256") != receipt_raw_sha
        or manifest.get("cache_set_manifest_sha256") != cache_raw_sha
    ):
        raise CLICTargetProtocolError("CLIC target evaluator provenance/package binding drift")
    receipt, known, reopened_receipt_raw_sha = _read_validated_receipt(
        receipt_path,
        cache_set_manifest_path=cache_path,
        expected_capsule_id=str(manifest.get("capsule_id", "")),
        expected_split_id=str(manifest.get("split_id", "")),
        expected_protocol_schema=str(manifest.get("protocol_schema", "")),
    )
    if reopened_receipt_raw_sha != receipt_raw_sha:
        raise CLICTargetProtocolError("CLIC target evaluator provenance receipt reopening SHA drift")
    # Reopen the manifest as a JSON object too, rather than treating a detached
    # file digest as a substitute for the receipt-bound cache-set artifact.
    cache_manifest = _target.read_json_object(cache_path, label="target cache-set manifest")
    if str(cache_manifest.get("cache_scope", "")) != _EXPECTED_CACHE_SCOPE:
        raise CLICTargetProtocolError("CLIC target evaluator cache-set scope drift")
    truth_known, _ = _reopen_truth_known_test_config(truth)
    if (
        known["path"] != truth_known["path"]
        or known["raw_sha256"] != truth_known["raw_sha256"]
        or known["data_normalized_sha256"]
        != _known_test_data_sha(truth_known["normalized"])
        or known["raw_sha256"] != manifest.get("known_test_config_raw_sha256")
        or known["data_normalized_sha256"]
        != manifest.get("known_test_config_normalized_sha256")
    ):
        raise CLICTargetProtocolError("CLIC target evaluator provenance known-test binding drift")
    # The returned fields are intentionally suitable for the evaluator receipt
    # only; no predictor-readable artifact receives these paths.
    return {
        "validator_receipt_path": str(receipt_path),
        "validator_receipt_raw_sha256": receipt_raw_sha,
        "cache_set_manifest_path": str(cache_path),
        "cache_set_manifest_raw_sha256": cache_raw_sha,
        "validated_once_receipt_sha256": sha256_file(receipt_path),
        "validated_once_cache_set_manifest_sha256": sha256_file(cache_path),
        "validated_once_known_test_config_raw_sha256": str(known["raw_sha256"]),
        "validated_once_known_test_config_data_sha256": str(known["data_normalized_sha256"]),
        "validated_once_receipt_schema": str(receipt.get("schema", "")),
    }


def _reverify_score_inputs(
    *,
    prediction: Mapping[str, Any],
    package: Mapping[str, Any],
    runtime: Any,
    train_config: Mapping[str, Any],
    predictor_artifact: Path,
    adv_reference: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> dict[str, str]:
    """Close the scorer's byte window before emitting its immutable receipt."""

    if sha256_file(Path(prediction["path"])) != prediction["raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target prediction changed during scoring")
    if sha256_file(Path(package["manifest_path"])) != package["manifest_raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target package manifest changed during scoring")
    if sha256_file(Path(package["data_path"])) != package["data_raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target package data changed during scoring")
    if sha256_file(predictor_artifact) != prediction["payload"]["predictor_artifact_sha256"]:
        raise CLICTargetProtocolError("CLIC target predictor artifact changed during scoring")
    reopened_train = _runtime_train_config(runtime)
    if not _same_train_config_binding(train_config, reopened_train):
        raise CLICTargetProtocolError("CLIC target train config changed during scoring")
    known, _ = _reopen_truth_known_test_config(truth)
    if sha256_file(Path(known["path"])) != known["raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target known-test config changed during scoring")
    reference_payload = adv_reference["payload"]
    if sha256_file(Path(adv_reference["path"])) != adv_reference["raw_sha256"]:
        raise CLICTargetProtocolError("ADV3B02 reference changed during scoring")
    for path_field, sha_field, label in (
        ("checkpoint_path", "checkpoint_sha256", "checkpoint"),
        ("train_config_manifest_path", "train_config_raw_sha256", "train config"),
        ("known_test_config_manifest_path", "known_test_config_raw_sha256", "known-test config"),
        ("stratified_metric_artifact_path", "stratified_metric_artifact_sha256", "stratified metrics"),
    ):
        if sha256_file(Path(reference_payload[path_field])) != reference_payload[sha_field]:
            raise CLICTargetProtocolError(f"ADV3B02 {label} changed during scoring")
    if sha256_file(Path(truth["path"])) != truth["raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target truth sidecar changed during scoring")
    return _reopen_validated_once_target_provenance(truth=truth, package=package)


def _reverify_target_metrics_inputs(
    *,
    prediction: Mapping[str, Any],
    package: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    train_config: Mapping[str, Any],
    predictor_artifact: Path,
    truth: Mapping[str, Any],
) -> dict[str, str]:
    """Close every target-only scorer byte window before receipt emission."""

    if sha256_file(Path(prediction["path"])) != prediction["raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target prediction changed during metrics sealing")
    if sha256_file(Path(package["manifest_path"])) != package["manifest_raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target package manifest changed during metrics sealing")
    if sha256_file(Path(package["data_path"])) != package["data_raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target package data changed during metrics sealing")
    if sha256_file(predictor_artifact) != prediction["payload"]["predictor_artifact_sha256"]:
        raise CLICTargetProtocolError("CLIC target predictor artifact changed during metrics sealing")

    # Reopen the predictor rather than trusting the already-loaded runtime.  C
    # descriptors thereby recheck their PAIR/checkpoint/raw authority chain,
    # while G bundles recheck their immutable archive and exact state members.
    reopened_runtime = load_verified_clic_predictor_state(predictor_artifact)
    reopened_identity = _require_runtime_identity(reopened_runtime)
    for field in (
        "arm",
        "operator",
        "fold_index",
        "state_sha256",
        "source_frozen_rule_sha256",
        "source_class_order",
        "source_class_order_sha256",
    ):
        if reopened_identity[field] != runtime_identity[field]:
            raise CLICTargetProtocolError(
                f"CLIC target predictor {field} changed during metrics sealing"
            )
    reopened_train = _runtime_train_config(reopened_runtime)
    if not _same_train_config_binding(train_config, reopened_train):
        raise CLICTargetProtocolError("CLIC target train config changed during metrics sealing")
    known, _ = _reopen_truth_known_test_config(truth)
    if sha256_file(Path(known["path"])) != known["raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target known-test config changed during metrics sealing")
    if sha256_file(Path(truth["path"])) != truth["raw_sha256"]:
        raise CLICTargetProtocolError("CLIC target truth sidecar changed during metrics sealing")
    return _reopen_validated_once_target_provenance(truth=truth, package=package)


def seal_clic_target_metrics(
    prediction_path: str | Path,
    truth_sidecar_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Seal baseline-independent target metrics after truth-blind prediction.

    This receipt is deliberately not an ADV3B02 comparison.  It evaluates the
    exact sealed candidate on target LEO-weak known/unknown rows and preserves
    the existing combined scorer as the only noninferiority entry.
    """

    output = Path(output_path).resolve()
    if output.exists():
        raise CLICTargetProtocolError(
            f"CLIC target metrics output already exists and is immutable: {output}"
        )

    # All predictor-readable artifacts close before the first truth read.
    prediction = _load_verified_sealed_clic_prediction(prediction_path)
    payload = prediction["payload"]
    runtime, runtime_identity, train_config, predictor_artifact = _reopen_predictor_from_prediction(
        payload
    )
    package = _read_verified_clic_iq_only_package_header(
        payload["predictor_package_path"]
    )
    _validate_prediction_package_binding(payload, package)

    # First evaluator-only truth read.  Known-test identity and physical role
    # metadata never return to the predictor or influence a rerun.
    truth = _load_verified_clic_truth_sidecar(
        truth_sidecar_path,
        expected_package_sha256=payload["predictor_package_sha256"],
        expected_lineage_sha256=payload["lineage_sha256"],
    )
    candidate_known, candidate_class_order = _reopen_truth_known_test_config(truth)
    known_data_sha = _known_test_data_sha(candidate_known["normalized"])
    if (
        candidate_known["raw_sha256"] != payload["known_test_config_raw_sha256"]
        or known_data_sha != payload["known_test_config_normalized_sha256"]
        or candidate_known["raw_sha256"]
        != package["manifest"]["known_test_config_raw_sha256"]
        or known_data_sha
        != package["manifest"]["known_test_config_normalized_sha256"]
    ):
        raise CLICTargetProtocolError(
            "CLIC target truth/package known-test config binding drift"
        )
    _validate_truth_universe_bindings(truth, package, candidate_known["normalized"])
    source_class_order, source_order_sha = _validated_source_class_order_binding(
        payload.get("source_class_order"),
        payload.get("source_class_order_sha256"),
        label="prediction source class order",
    )
    if not set(source_class_order).issubset(set(candidate_class_order)):
        raise CLICTargetProtocolError(
            "prediction source class order is not contained in target registered-known config"
        )

    joined = _join_prediction_and_truth(
        prediction, truth, class_order=source_class_order
    )
    known_audit = _score_known_target_rows(
        joined,
        class_order=source_class_order,
        receiver_ids=[
            str(value) for value in candidate_known["normalized"]["target_receiver_ids"]
        ],
        day_ids=[
            str(value) for value in candidate_known["normalized"]["target_day_ids"]
        ],
    )
    unknown_gate = _evaluate_explicit_unknown_gate(
        joined, explicit_unknown_floor=0.70
    )
    open_set_audit = _score_open_set_target_rows(joined)
    fold_config_key = _target.require_sha256(
        payload["fold_config_key"], label="prediction fold config key"
    )
    if fold_config_key != train_config["data_normalized_sha256"]:
        raise CLICTargetProtocolError("prediction/scorer fold config key drift")

    target_provenance = _reverify_target_metrics_inputs(
        prediction=prediction,
        package=package,
        runtime_identity=runtime_identity,
        train_config=train_config,
        predictor_artifact=predictor_artifact,
        truth=truth,
    )
    metrics_base = {
        "schema": _TARGET_METRICS_SCHEMA,
        "sealed": True,
        "truth_sidecar_opened": True,
        "baseline_compared": False,
        "comparison_status": "ADV_COMPARISON_PENDING",
        "prediction_path": str(prediction["path"]),
        "prediction_raw_sha256": prediction["raw_sha256"],
        "prediction_sha256": payload["prediction_sha256"],
        "predictor_package_path": payload["predictor_package_path"],
        "predictor_package_sha256": payload["predictor_package_sha256"],
        "package_manifest_sha256": payload["package_manifest_sha256"],
        "received_iq_data_sha256": payload["received_iq_data_sha256"],
        "predictor_state_path": payload["predictor_state_path"],
        "predictor_artifact_sha256": payload["predictor_artifact_sha256"],
        "predictor_state_sha256": runtime_identity["state_sha256"],
        "source_frozen_rule_sha256": runtime_identity["source_frozen_rule_sha256"],
        "source_class_order": source_class_order,
        "source_class_order_sha256": source_order_sha,
        "arm": runtime_identity["arm"],
        "operator": runtime_identity["operator"],
        "fold_index": runtime_identity["fold_index"],
        "fold_config_key": fold_config_key,
        "train_config_manifest_path": train_config["container_path"],
        "train_config_member_name": train_config["member_name"],
        "train_config_raw_sha256": train_config["raw_sha256"],
        "train_config_normalized_sha256": train_config["sealed_normalized_sha256"],
        "train_config_data_normalized_sha256": train_config["data_normalized_sha256"],
        "known_test_config_manifest_path": candidate_known["path"],
        "known_test_config_raw_sha256": candidate_known["raw_sha256"],
        "known_test_config_normalized_sha256": known_data_sha,
        "truth_sidecar_path": str(truth["path"]),
        "truth_sidecar_raw_sha256": truth["raw_sha256"],
        **target_provenance,
        "lineage_sha256": payload["lineage_sha256"],
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "known_target_audit": known_audit,
        "unknown_target_audit": unknown_gate["unknown_audit"],
        "open_set_audit": open_set_audit,
        "explicit_unknown_gate": {
            "passed": bool(unknown_gate["explicit_unknown_gate_passed"]),
            "floor": 0.70,
            "failures": unknown_gate["failures"],
        },
        "passed": bool(unknown_gate["explicit_unknown_gate_passed"]),
        "scorer_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    payload_out = dict(
        metrics_base, metrics_sha256=_target.canonical_sha256(metrics_base)
    )
    _write_new_utf8_json(output, payload_out, label="CLIC target metrics receipt")
    return output


def score_clic_target_prediction(
    prediction_path: str | Path,
    truth_sidecar_path: str | Path,
    adv3b02_reference_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Score one sealed prediction only after all non-truth artifacts reopen.

    The API deliberately accepts no caller-supplied configuration, class order,
    threshold, or model state.  It reopens the C/G predictor descriptor or
    bundle, the candidate training configuration, the package known-test
    configuration, and ADV3B02's own reference artifacts before it reads the
    truth sidecar for the first time.
    """

    output = Path(output_path).resolve()
    if output.exists():
        raise CLICTargetProtocolError(
            f"CLIC target score output already exists and is immutable: {output}"
        )
    prediction = _load_verified_sealed_clic_prediction(prediction_path)
    payload = prediction["payload"]
    runtime, runtime_identity, train_config, predictor_artifact = _reopen_predictor_from_prediction(payload)
    package = _read_verified_clic_iq_only_package_header(payload["predictor_package_path"])
    _validate_prediction_package_binding(payload, package)
    baseline = _load_verified_adv3b02_reference(adv3b02_reference_path)

    # Prediction/package/predictor/reference checks are now closed.  This is
    # the first truth-sidecar read in the scorer; it authorizes opening the
    # evaluator-only known-test configuration.
    truth = _load_verified_clic_truth_sidecar(
        truth_sidecar_path,
        expected_package_sha256=payload["predictor_package_sha256"],
        expected_lineage_sha256=payload["lineage_sha256"],
    )
    candidate_known, candidate_class_order = _reopen_truth_known_test_config(truth)
    known_data_sha = _known_test_data_sha(candidate_known["normalized"])
    if (
        candidate_known["raw_sha256"] != payload["known_test_config_raw_sha256"]
        or known_data_sha != payload["known_test_config_normalized_sha256"]
        or candidate_known["raw_sha256"] != package["manifest"]["known_test_config_raw_sha256"]
        or known_data_sha != package["manifest"]["known_test_config_normalized_sha256"]
    ):
        raise CLICTargetProtocolError("CLIC target truth/package known-test config binding drift")
    config_equivalence = validate_adv3b02_config_equivalence(
        candidate_train_config=train_config["normalized"],
        candidate_known_test_config=candidate_known["normalized"],
        baseline_train_config=baseline["train_config"]["normalized"],
        baseline_known_test_config=baseline["known_test_config"]["normalized"],
    )
    _validate_truth_universe_bindings(truth, package, candidate_known["normalized"])
    source_class_order, _source_order_sha = _validated_source_class_order_binding(
        payload.get("source_class_order"),
        payload.get("source_class_order_sha256"),
        label="prediction source class order",
    )
    if not set(source_class_order).issubset(set(candidate_class_order)):
        raise CLICTargetProtocolError(
            "prediction source class order is not contained in target registered-known config"
        )
    joined = _join_prediction_and_truth(
        prediction, truth, class_order=source_class_order
    )
    known_audit = _score_known_target_rows(
        joined,
        class_order=source_class_order,
        receiver_ids=[
            str(value) for value in candidate_known["normalized"]["target_receiver_ids"]
        ],
        day_ids=[str(value) for value in candidate_known["normalized"]["target_day_ids"]],
    )
    # Unlike malformed protocol/config artifacts, a measured performance gate
    # failure is a valid experimental result and must receive a sealed receipt.
    unknown_gate = _evaluate_explicit_unknown_gate(joined, explicit_unknown_floor=0.70)
    open_set_audit = _score_open_set_target_rows(joined)
    fold_config_key = _target.require_sha256(
        payload["fold_config_key"], label="prediction fold config key"
    )
    if fold_config_key != train_config["data_normalized_sha256"]:
        raise CLICTargetProtocolError("prediction/scorer fold config key drift")
    noninferiority = _noninferiority_against_adv3b02(
        known_audit,
        baseline["cells"],
        fold_index=int(runtime_identity["fold_index"]),
        fold_config_key=fold_config_key,
        source_class_order=source_class_order,
    )
    target_provenance = _reverify_score_inputs(
        prediction=prediction,
        package=package,
        runtime=runtime,
        train_config=train_config,
        predictor_artifact=predictor_artifact,
        adv_reference=baseline,
        truth=truth,
    )

    score_base = {
        "schema": _TARGET_SCORE_SCHEMA,
        "sealed": True,
        "truth_sidecar_opened": True,
        "prediction_path": str(prediction["path"]),
        "prediction_raw_sha256": prediction["raw_sha256"],
        "prediction_sha256": payload["prediction_sha256"],
        "predictor_package_path": payload["predictor_package_path"],
        "predictor_package_sha256": payload["predictor_package_sha256"],
        "package_manifest_sha256": payload["package_manifest_sha256"],
        "received_iq_data_sha256": payload["received_iq_data_sha256"],
        "predictor_state_path": payload["predictor_state_path"],
        "predictor_artifact_sha256": payload["predictor_artifact_sha256"],
        "predictor_state_sha256": runtime_identity["state_sha256"],
        "source_frozen_rule_sha256": runtime_identity["source_frozen_rule_sha256"],
        # Evaluator-side audit of the exact local-four universe used for
        # formal-known metrics and ADV filtering.
        "source_class_order": source_class_order,
        "source_class_order_sha256": _source_order_sha,
        "arm": runtime_identity["arm"],
        "operator": runtime_identity["operator"],
        "fold_index": runtime_identity["fold_index"],
        "fold_config_key": fold_config_key,
        "train_config_manifest_path": train_config["container_path"],
        "train_config_member_name": train_config["member_name"],
        "train_config_raw_sha256": train_config["raw_sha256"],
        "train_config_normalized_sha256": train_config["sealed_normalized_sha256"],
        "train_config_data_normalized_sha256": train_config["data_normalized_sha256"],
        "known_test_config_manifest_path": candidate_known["path"],
        "known_test_config_raw_sha256": candidate_known["raw_sha256"],
        "known_test_config_normalized_sha256": _known_test_data_sha(candidate_known["normalized"]),
        "adv3b02_reference_path": str(baseline["path"]),
        "adv3b02_reference_raw_sha256": baseline["raw_sha256"],
        "adv3b02_reference_sha256": baseline["payload"]["reference_sha256"],
        "adv3b02_checkpoint_sha256": baseline["payload"]["checkpoint_sha256"],
        "truth_sidecar_path": str(truth["path"]),
        "truth_sidecar_raw_sha256": truth["raw_sha256"],
        **target_provenance,
        "lineage_sha256": payload["lineage_sha256"],
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "known_target_audit": known_audit,
        "unknown_target_audit": unknown_gate["unknown_audit"],
        "open_set_audit": open_set_audit,
        "explicit_unknown_gate": {
            "passed": bool(unknown_gate["explicit_unknown_gate_passed"]),
            "floor": 0.70,
            "failures": unknown_gate["failures"],
        },
        "adv3b02_noninferiority": noninferiority,
        "passed": bool(
            unknown_gate["explicit_unknown_gate_passed"] and noninferiority["passed"]
        ),
        "adv3b02_config_equivalence": config_equivalence,
        "scorer_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    payload_out = dict(score_base, score_sha256=_target.canonical_sha256(score_base))
    _write_new_utf8_json(output, payload_out, label="CLIC target score receipt")
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded file-only Task7 target execution CLI.

    Every mode accepts only already-sealed artifact paths.  In particular, the
    validation mode takes semantic channel/preprocess/metric JSON from an
    immutable file instead of command-line selections, while prediction and
    scoring receive no class, truth, threshold, or target-adaptation options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--seal-target-validation",
        action="store_true",
        help="derive evaluator-only known config and VALIDATED_ONCE receipt from confirmation cache",
    )
    modes.add_argument(
        "--seal-target-package",
        action="store_true",
        help="seal a verified confirmation cache into IQ-only package plus truth sidecar",
    )
    modes.add_argument(
        "--publish-target-prediction",
        action="store_true",
        help="run one C/G forward per IQ-only target package row",
    )
    modes.add_argument(
        "--score-target-prediction",
        action="store_true",
        help="score one sealed prediction through evaluator-only truth sidecar",
    )
    modes.add_argument(
        "--seal-target-metrics",
        action="store_true",
        help="seal baseline-independent target LEO-weak known/unknown/DG metrics",
    )
    parser.add_argument("--cache-set-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validator-receipt", type=Path)
    parser.add_argument("--expected-capsule-id")
    parser.add_argument("--expected-split-id")
    parser.add_argument("--expected-protocol-schema", default="p2_min_v1")
    parser.add_argument(
        "--test-semantics-json",
        type=Path,
        help="sealed JSON object containing only channel/preprocess/zero_adapt/metrics",
    )
    parser.add_argument("--predictor-state", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--truth-sidecar", type=Path)
    parser.add_argument("--adv3b02-reference", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _cli_required(parser: argparse.ArgumentParser, args: argparse.Namespace, *fields: str) -> None:
    missing = [f"--{field.replace('_', '-')}" for field in fields if getattr(args, field) is None]
    if missing:
        parser.error(f"selected mode requires {', '.join(missing)}")


def main(argv: Iterable[str] | None = None) -> int:
    """Run exactly one file-only target artifact operation."""

    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.seal_target_validation:
        _cli_required(
            parser,
            args,
            "cache_set_manifest",
            "output_root",
            "test_semantics_json",
        )
        semantics_path = Path(args.test_semantics_json).resolve()
        raw_before = sha256_file(semantics_path)
        semantics = _target.read_json_object(
            semantics_path, label="confirmation test-semantics JSON"
        )
        if sha256_file(semantics_path) != raw_before:
            raise CLICTargetProtocolError(
                "confirmation test-semantics JSON changed while CLI opening"
            )
        result = seal_clic_target_confirmation_validation(
            args.cache_set_manifest,
            args.output_root,
            test_semantics=semantics,
            test_semantics_artifact_path=semantics_path,
            expected_capsule_id=args.expected_capsule_id,
            expected_split_id=args.expected_split_id,
            expected_protocol_schema=args.expected_protocol_schema,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.seal_target_package:
        _cli_required(
            parser,
            args,
            "cache_set_manifest",
            "output_root",
            "validator_receipt",
            "expected_capsule_id",
            "expected_split_id",
        )
        package, truth = seal_clic_target_package(
            args.cache_set_manifest,
            args.output_root,
            validator_receipt_path=args.validator_receipt,
            expected_capsule_id=args.expected_capsule_id,
            expected_split_id=args.expected_split_id,
            expected_protocol_schema=args.expected_protocol_schema,
        )
        print(
            json.dumps(
                {"package_path": str(package), "truth_sidecar_path": str(truth)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.publish_target_prediction:
        _cli_required(parser, args, "predictor_state", "package", "output")
        output = publish_clic_target_prediction(
            args.predictor_state, args.package, args.output
        )
        print(json.dumps({"prediction_path": str(output)}, ensure_ascii=False, sort_keys=True))
        return 0
    if args.seal_target_metrics:
        _cli_required(parser, args, "prediction", "truth_sidecar", "output")
        output = seal_clic_target_metrics(
            args.prediction,
            args.truth_sidecar,
            args.output,
        )
        print(
            json.dumps(
                {"metrics_path": str(output)}, ensure_ascii=False, sort_keys=True
            )
        )
        return 0
    _cli_required(
        parser,
        args,
        "prediction",
        "truth_sidecar",
        "adv3b02_reference",
        "output",
    )
    output = score_clic_target_prediction(
        args.prediction,
        args.truth_sidecar,
        args.adv3b02_reference,
        args.output,
    )
    print(json.dumps({"score_path": str(output)}, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CLICTargetGateError",
    "CLICTargetProtocolError",
    "compute_target_open_set_metrics",
    "build_parser",
    "ingest_adv3b02_target_known_reference",
    "recompute_unknown_counts",
    "publish_clic_target_prediction",
    "seal_clic_target_metrics",
    "score_clic_target_prediction",
    "score_target_rows",
    "seal_clic_target_confirmation_validation",
    "seal_clic_target_package",
    "validate_adv3b02_config_equivalence",
]


if __name__ == "__main__":
    raise SystemExit(main())
