from __future__ import annotations

import ast
import hashlib
import inspect
import json
import copy
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.somph_predictor_bundle as somph_bundle
import cvsrffi.somph_predictor_runtime as somph_runtime
from cvsrffi.somph_diagnostic_bundle_loader import (
    load_verified_somph_predictor_bundle,
    preflight_somph_predictor_bundle,
)
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_predictor_bundle import (
    ADV3B02_FEATURE_SCHEMA,
    APPLY_ONLY,
    ENROLLMENT_ONLY,
    FORMAL_LEO_WEAK_SCENARIOS,
    HEAD_CAPSULE_NPZ_MEMBERS,
    QUERY_NPZ_MEMBERS,
    SOMPH_METHOD_LOCK_SCHEMA,
    SOMPH_OVERLAY_PROVENANCE_SCHEMA,
    SOMPH_QUERY_IQ_SCHEMA,
    SOMPH_SUPPORT_IQ_SCHEMA,
    SUPPORT_NPZ_MEMBERS,
    PredictorPackageError,
    iq_row_sha256,
    sha256_file,
    write_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_runtime import (
    SOMPH_ENROLLMENT_BINDING_SCHEMA,
    expected_somph_method_lock,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _method_lock(path: Path, checkpoint_sha256: str) -> None:
    payload = expected_somph_method_lock()
    payload["checkpoint_sha256"] = checkpoint_sha256
    path.write_bytes(somph_bundle.canonical_json_bytes(payload))


def _token(prefix: str, index: int) -> str:
    return prefix + f"{index:064x}"


def _valid_head_capsule(
    *,
    binding: dict,
    method_sha: str,
    class_count: int,
    k_shot: int,
) -> dict[str, np.ndarray]:
    centroids = np.eye(class_count, 160, dtype=np.float16)
    prototypes = np.repeat(centroids, min(2, k_shot), axis=0)
    prototype_ids = np.repeat(
        np.arange(class_count, dtype=np.uint16), min(2, k_shot)
    )
    payload: dict[str, np.ndarray] = {
        "schema_utf8": np.frombuffer(
            b"cvs.phase2.somph_runtime_head_capsule.v1", dtype=np.uint8
        ).copy(),
        "method_lock_sha256_utf8": np.frombuffer(
            method_sha.encode("ascii"), dtype=np.uint8
        ).copy(),
        "enrollment_binding_json_utf8": np.frombuffer(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
            dtype=np.uint8,
        ).copy(),
        "class_count_uint16": np.asarray([class_count], dtype=np.uint16),
        "feature_dim_uint16": np.asarray([160], dtype=np.uint16),
        "k_shot_uint16": np.asarray([k_shot], dtype=np.uint16),
    }
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload[f"{scenario}__prototypes_fp16"] = prototypes.copy()
        payload[f"{scenario}__prototype_class_ids_uint16"] = prototype_ids.copy()
        payload[f"{scenario}__centroids_fp16"] = centroids.copy()
        payload[f"{scenario}__residual_scale_fp16"] = np.ones(
            160, dtype=np.float16
        )
        payload[f"{scenario}__class_hubness_penalty_fp16"] = np.zeros(
            class_count, dtype=np.float16
        )
        payload[f"{scenario}__scalars_fp16"] = np.asarray(
            [0.75, 0.25], dtype=np.float16
        )
    assert tuple(payload) == HEAD_CAPSULE_NPZ_MEMBERS
    return payload


def _support_npz(
    path: Path,
    *,
    scenario: str,
    class_count: int,
    declared_k: int,
    registration_state: str,
    sample_token_offset: int = 0,
    support_rows_per_class: int | None = None,
    corrupt_first_assignment: bool = False,
) -> list[dict]:
    rows_per_class = (
        declared_k if support_rows_per_class is None else support_rows_per_class
    )
    count = class_count * rows_per_class
    iq = np.arange(count * 2 * 4, dtype=np.float32).reshape(count, 2, 4)
    tokens = [
        _token("sid_", sample_token_offset + index + 1) for index in range(count)
    ]
    scenario_offset = FORMAL_LEO_WEAK_SCENARIOS.index(scenario) * 1000
    overlays = [
        _token("oid_", 10000 + scenario_offset + index) for index in range(count)
    ]
    seeds = np.arange(
        20000 + scenario_offset, 20000 + scenario_offset + count, dtype=np.int64
    )
    hashes = [iq_row_sha256(row) for row in iq]
    labels = np.repeat(
        np.arange(class_count, dtype=np.int64), rows_per_class
    )
    ranks = np.tile(np.arange(rows_per_class, dtype=np.int64), class_count)
    if corrupt_first_assignment:
        ranks[0] = rows_per_class - 1
    embedded = {
        "schema": SOMPH_SUPPORT_IQ_SCHEMA,
        "scenario": scenario,
        "registration_state": registration_state,
        "registered_class_count": class_count,
        "support_pool_max_k": declared_k,
        "token_scheme": "hmac_sha256_opaque_v1",
    }
    with path.open("xb") as handle:
        np.savez(
            handle,
            support_leo_weak_iq=iq,
            support_class_indices=labels,
            support_rank_within_class=ranks,
            support_tokens=np.asarray(tokens),
            support_overlay_tokens=np.asarray(overlays),
            support_satellite_seeds=seeds,
            support_post_channel_iq_sha256=np.asarray(hashes),
            manifest_json=np.asarray(json.dumps(embedded, sort_keys=True)),
        )
    return [
        {
            "sample_token": token,
            "scenario": scenario,
            "overlay_token": overlay,
            "satellite_seed": int(seed),
            "post_channel_iq_sha256": digest,
            "source_leo_cache_sha256": "a" * 64,
            "source_leo_provenance_sha256": "b" * 64,
        }
        for token, overlay, seed, digest in zip(tokens, overlays, seeds, hashes)
    ]


def _query_npz(
    path: Path,
    *,
    scenario: str,
    registration_state: str,
    sample_token_offset: int = 0,
) -> list[dict]:
    iq = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    scenario_offset = FORMAL_LEO_WEAK_SCENARIOS.index(scenario) * 100
    tokens = [
        _token("qid_", sample_token_offset + index + 1) for index in range(3)
    ]
    overlays = [_token("oid_", 30000 + scenario_offset + index) for index in range(3)]
    seeds = np.arange(40000 + scenario_offset, 40003 + scenario_offset, dtype=np.int64)
    hashes = [iq_row_sha256(row) for row in iq]
    embedded = {
        "schema": SOMPH_QUERY_IQ_SCHEMA,
        "scenario": scenario,
        "registration_state": registration_state,
        "token_scheme": "hmac_sha256_opaque_v1",
    }
    with path.open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=iq,
            query_tokens=np.asarray(tokens),
            query_overlay_tokens=np.asarray(overlays),
            query_satellite_seeds=seeds,
            query_post_channel_iq_sha256=np.asarray(hashes),
            manifest_json=np.asarray(json.dumps(embedded, sort_keys=True)),
        )
    return [
        {
            "sample_token": token,
            "scenario": scenario,
            "overlay_token": overlay,
            "satellite_seed": int(seed),
            "post_channel_iq_sha256": digest,
            "source_leo_cache_sha256": "c" * 64,
            "source_leo_provenance_sha256": "d" * 64,
        }
        for token, overlay, seed, digest in zip(tokens, overlays, seeds, hashes)
    ]


def _package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str,
    stage: str = "stage2c",
    registration_state: str = "before",
    class_count: int = 2,
    k_shot: int = 5,
    drift_last_scenario_tokens: bool = False,
    noncanonical_method_lock: bool = False,
    drift_last_support_assignment: bool = False,
    invalid_head_scalars: bool = False,
    head_trust_root_override: str | None = None,
    overlay_trust_root_override: str | None = None,
    support_rows_per_class: int | None = None,
    declared_support_pool_max_k: int | None = None,
    row_handle: str = "row_" + "4" * 64,
    row_manifest_sha256: str = "5" * 64,
    receiver: str = "20-1",
):
    root = tmp_path / profile
    root.mkdir(parents=True)
    feature_runtime = root / somph_bundle.FEATURE_RUNTIME_RELATIVE_PATH
    feature_runtime.write_bytes(b"synthetic-adv3b02-torchscript-runtime")
    feature_runtime_sha = sha256_file(feature_runtime)
    phase1_checkpoint_sha = "e" * 64
    monkeypatch.setattr(
        somph_bundle,
        "ADV3B02_PHASE1_CHECKPOINT_SHA256",
        phase1_checkpoint_sha,
    )
    monkeypatch.setattr(
        somph_runtime,
        "ADV3B02_PHASE1_CHECKPOINT_SHA256",
        phase1_checkpoint_sha,
    )
    _method_lock(root / "method_lock.json", phase1_checkpoint_sha)
    if noncanonical_method_lock:
        payload = json.loads((root / "method_lock.json").read_text(encoding="utf-8"))
        (root / "method_lock.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    registry = [
        {"class_index": index, "class_handle": _token("cls_", index + 1)}
        for index in range(class_count)
    ]
    method_sha = sha256_file(root / "method_lock.json")
    head_binding_sha = None
    head_capsule_sha = None
    if profile == APPLY_ONLY:
        head_binding = {
            "schema": SOMPH_ENROLLMENT_BINDING_SCHEMA,
            "stage": stage,
            "registration_state": registration_state,
            "receiver": receiver,
            "seed": 713101,
            "k_shot": k_shot,
            "registered_class_handles": [
                item["class_handle"] for item in registry
            ],
            "enrollment_package_root_sha256": "1" * 64,
            "enrollment_package_seal_sha256": "2" * 64,
            "phase1_checkpoint_sha256": phase1_checkpoint_sha,
            "feature_runtime_sha256": feature_runtime_sha,
            "method_lock_sha256": method_sha,
            "support_token_sha256_by_scenario": {
                scenario: "3" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
            "support_feature_sha256_by_scenario": {
                scenario: "4" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
        }
        head_binding_sha = somph_bundle.sha256_bytes(
            somph_bundle.canonical_json_bytes(head_binding)
        )
        with (root / "head_capsule.npz").open("xb") as handle:
            head_payload = _valid_head_capsule(
                binding=head_binding,
                method_sha=method_sha,
                class_count=class_count,
                k_shot=k_shot,
            )
            if invalid_head_scalars:
                head_payload[
                    f"{FORMAL_LEO_WEAK_SCENARIOS[0]}__scalars_fp16"
                ] = np.asarray([0.5, 0.25], dtype=np.float16)
            np.savez(
                handle,
                **head_payload,
            )
        head_capsule_sha = sha256_file(root / "head_capsule.npz")
    samples = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        scenario_index = FORMAL_LEO_WEAK_SCENARIOS.index(scenario)
        sample_token_offset = scenario_index * 1000
        if (
            drift_last_scenario_tokens
            and scenario == FORMAL_LEO_WEAK_SCENARIOS[-1]
        ):
            sample_token_offset = 0
        if profile == ENROLLMENT_ONLY:
            samples.extend(
                _support_npz(
                    root / f"support_{scenario}.npz",
                    scenario=scenario,
                    class_count=class_count,
                    declared_k=k_shot,
                    registration_state=registration_state,
                    sample_token_offset=sample_token_offset,
                    support_rows_per_class=support_rows_per_class,
                    corrupt_first_assignment=(
                        drift_last_support_assignment
                        and scenario == FORMAL_LEO_WEAK_SCENARIOS[-1]
                    ),
                )
            )
        else:
            samples.extend(
                _query_npz(
                    root / f"query_{scenario}.npz",
                    scenario=scenario,
                    registration_state=registration_state,
                    sample_token_offset=sample_token_offset,
                )
            )
    _write_json(
        root / "overlay_provenance.json",
        {
            "schema": SOMPH_OVERLAY_PROVENANCE_SCHEMA,
            "profile": profile,
            "receiver": receiver,
            "seed": 713101,
            "samples": samples,
        },
    )
    seal = tmp_path / f"{profile}.seal.json"
    overlay_provenance_sha = sha256_file(root / "overlay_provenance.json")
    _manifest_path, _seal_path, manifest, _seal = write_somph_predictor_bundle(
        root,
        profile=profile,
        stage=stage,
        registration_state=registration_state,
        receiver=receiver,
        seed=713101,
        k_shot=k_shot,
        registered_classes=registry,
        expected_method_lock_sha256=method_sha,
        expected_overlay_provenance_sha256=(
            overlay_trust_root_override or overlay_provenance_sha
        ),
        detached_seal_path=seal,
        expected_head_enrollment_binding_sha256=head_binding_sha,
        expected_head_capsule_sha256=(
            head_trust_root_override or head_capsule_sha
        ),
        expected_row_handle=row_handle if profile == APPLY_ONLY else None,
        expected_row_manifest_sha256=(
            row_manifest_sha256 if profile == APPLY_ONLY else None
        ),
        support_pool_max_k=declared_support_pool_max_k,
    )
    return root, seal, sha256_file(seal), manifest


@pytest.mark.parametrize(
    ("profile", "expected", "forbidden"),
    [
        (
            ENROLLMENT_ONLY,
            {"feature_runtime", "method_lock", "overlay_provenance"}
            | {f"support:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS},
            {"head_capsule"} | {f"query:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS},
        ),
        (
            APPLY_ONLY,
            {
                "feature_runtime",
                "method_lock",
                "head_capsule",
                "overlay_provenance",
            }
            | {f"query:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS},
            {f"support:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS},
        ),
    ],
)
def test_profiles_are_physically_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected: set[str],
    forbidden: set[str],
) -> None:
    root, seal, digest, manifest = _package(tmp_path, monkeypatch, profile=profile)
    assert {item["kind"] for item in manifest["members"]} == expected
    assert not ({item["kind"] for item in manifest["members"]} & forbidden)
    _manifest, _seal, audit = preflight_somph_predictor_bundle(
        root, detached_seal_path=seal, expected_seal_sha256=digest
    )
    assert audit["status"] == "UNVERIFIED_UNDER_CURRENT_PROTOCOL_DIAGNOSTIC_ONLY"
    assert audit["diagnostic_only"] is True
    assert audit["formal_launch_authority"] is False
    assert audit["iq_payload_materialized"] is False


def test_manifest_has_no_old_new_boundary_or_query_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, _seal, _digest, manifest = _package(
        tmp_path, monkeypatch, profile=APPLY_ONLY, registration_state="after"
    )
    text = json.dumps(manifest, sort_keys=True).lower()
    assert "new_class_count" not in text
    assert "old_class" not in text
    assert "new_class" not in text
    assert "q20" not in text
    assert "query_order" not in text
    assert "query_truth" not in text
    # Required false-valued Phase2 contract fields are the only role/quota surface.
    assert manifest["phase2_query_role_oracle_access"] is False
    assert manifest["phase2_query_class_quota_access"] is False
    assert manifest["row_handle"] == "row_" + "4" * 64
    assert manifest["row_manifest_sha256"] == "5" * 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("row_handle", "row_bad", "opaque row handle"),
        ("row_manifest_sha256", "F" * 64, "strict row manifest SHA256"),
    ],
)
def test_apply_sealing_requires_strict_row_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    kwargs = {field: value}
    with pytest.raises(PredictorPackageError, match=message):
        _package(
            tmp_path,
            monkeypatch,
            profile=APPLY_ONLY,
            **kwargs,
        )


def test_noncanonical_method_lock_bytes_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        PredictorPackageError,
        match="method lock must use canonical JSON bytes",
    ):
        _package(
            tmp_path,
            monkeypatch,
            profile=ENROLLMENT_ONLY,
            noncanonical_method_lock=True,
        )


def test_apply_head_must_match_external_enrollment_capsule_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        PredictorPackageError,
        match="does not match enrollment trust root",
    ):
        _package(
            tmp_path,
            monkeypatch,
            profile=APPLY_ONLY,
            head_trust_root_override="f" * 64,
        )


def test_overlay_provenance_must_match_external_exporter_trust_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        PredictorPackageError,
        match="overlay provenance does not match external trust root",
    ):
        _package(
            tmp_path,
            monkeypatch,
            profile=ENROLLMENT_ONLY,
            overlay_trust_root_override="f" * 64,
        )


def test_apply_head_semantics_fail_before_query_iq_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspected = []
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *args, **kwargs: inspected.append(True),
    )
    with pytest.raises(
        PredictorPackageError,
        match="head semantic validation failed",
    ):
        _package(
            tmp_path,
            monkeypatch,
            profile=APPLY_ONLY,
            invalid_head_scalars=True,
        )
    assert inspected == []


def test_wrong_adv3b02_phase1_lineage_fails_before_iq_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, digest, _manifest = _package(
        tmp_path, monkeypatch, profile=APPLY_ONLY
    )
    monkeypatch.setattr(
        somph_bundle, "ADV3B02_PHASE1_CHECKPOINT_SHA256", "f" * 64
    )
    inspected = []
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *args, **kwargs: inspected.append(True),
    )
    with pytest.raises(PredictorPackageError, match="before IQ|contract failed"):
        preflight_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )
    assert inspected == []


def test_provenance_tamper_fails_before_iq_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, digest, _manifest = _package(
        tmp_path, monkeypatch, profile=ENROLLMENT_ONLY
    )
    provenance = root / "overlay_provenance.json"
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["samples"][0]["satellite_seed"] += 1
    _write_json(provenance, payload)
    inspected = []
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *args, **kwargs: inspected.append(True),
    )
    with pytest.raises(PredictorPackageError, match="digest mismatch"):
        preflight_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )
    assert inspected == []


def test_extra_member_and_extra_root_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, digest, _manifest = _package(
        tmp_path, monkeypatch, profile=APPLY_ONLY
    )
    query = root / f"query_{FORMAL_LEO_WEAK_SCENARIOS[0]}.npz"
    with np.load(query, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["debug"] = np.asarray([1])
    query.unlink()
    with query.open("xb") as handle:
        np.savez(handle, **payload)
    with pytest.raises(PredictorPackageError, match="digest mismatch|allowlist"):
        preflight_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )

    root2, seal2, digest2, _manifest2 = _package(
        tmp_path / "second", monkeypatch, profile=APPLY_ONLY
    )
    (root2 / "debug.txt").write_text("no", encoding="utf-8")
    with pytest.raises(PredictorPackageError, match="unexpected package file"):
        preflight_somph_predictor_bundle(
            root2, detached_seal_path=seal2, expected_seal_sha256=digest2
        )


@pytest.mark.parametrize("forbidden", ["query_truth", "query_role", "query_quota"])
def test_query_npz_forbids_truth_role_and_quota_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbidden: str
) -> None:
    root, seal, digest, manifest = _package(
        tmp_path, monkeypatch, profile=APPLY_ONLY
    )
    query = root / f"query_{FORMAL_LEO_WEAK_SCENARIOS[0]}.npz"
    with np.load(query, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload[forbidden] = np.asarray([0])
    query.unlink()
    with query.open("xb") as handle:
        np.savez(handle, **payload)
    assert forbidden not in QUERY_NPZ_MEMBERS
    descriptor = next(
        dict(item)
        for item in manifest["members"]
        if item["kind"] == f"query:{FORMAL_LEO_WEAK_SCENARIOS[0]}"
    )
    descriptor["sha256"] = sha256_file(query)
    descriptor["size_bytes"] = query.stat().st_size
    with pytest.raises(PredictorPackageError, match="NPZ member allowlist mismatch"):
        somph_bundle._inspect_iq_member(root, descriptor)
    with pytest.raises(PredictorPackageError, match="digest mismatch|allowlist"):
        preflight_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_before_after_stage_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _package(
        tmp_path / "b",
        monkeypatch,
        profile=APPLY_ONLY,
        stage="stage2b",
        registration_state="before",
    )
    _package(
        tmp_path / "c_before",
        monkeypatch,
        profile=APPLY_ONLY,
        stage="stage2c",
        registration_state="before",
    )
    _package(
        tmp_path / "c_after",
        monkeypatch,
        profile=APPLY_ONLY,
        stage="stage2c",
        registration_state="after",
    )
    with pytest.raises(PredictorPackageError, match="Stage2-B"):
        _package(
            tmp_path / "bad",
            monkeypatch,
            profile=APPLY_ONLY,
            stage="stage2b",
            registration_state="after",
        )


@pytest.mark.parametrize("profile", [ENROLLMENT_ONLY, APPLY_ONLY])
def test_sample_level_provenance_crosscheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    root, seal, digest, _manifest = _package(tmp_path, monkeypatch, profile=profile)
    payloads, manifest, audit = load_verified_somph_predictor_bundle(
        root, detached_seal_path=seal, expected_seal_sha256=digest
    )
    assert manifest["profile"] == profile
    assert set(payloads) == set(FORMAL_LEO_WEAK_SCENARIOS)
    assert audit["sample_level_overlay_provenance_crosscheck"] == "PASS"


def test_sample_level_provenance_mismatch_rejected_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, digest, _manifest = _package(
        tmp_path, monkeypatch, profile=APPLY_ONLY
    )
    original = somph_bundle._materialize_iq

    def drift(*args, **kwargs):
        arrays, embedded = original(*args, **kwargs)
        arrays["query_satellite_seeds"][0] += 1
        return arrays, embedded

    monkeypatch.setattr(somph_bundle, "_materialize_iq", drift)
    with pytest.raises(PredictorPackageError, match="sample-level provenance"):
        load_verified_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


@pytest.mark.parametrize("profile", [ENROLLMENT_ONLY, APPLY_ONLY])
def test_cross_scenario_physical_sample_token_reuse_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    root, seal, digest, _manifest = _package(
        tmp_path,
        monkeypatch,
        profile=profile,
        drift_last_scenario_tokens=True,
    )
    with pytest.raises(
        PredictorPackageError,
        match="physical sample-token reuse across LEO_weak scenarios",
    ):
        load_verified_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_each_scenario_support_structure_must_be_exact_ordered_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        PredictorPackageError,
        match="exactly K ordered samples",
    ):
        _package(
            tmp_path,
            monkeypatch,
            profile=ENROLLMENT_ONLY,
            drift_last_support_assignment=True,
        )
    assert not (tmp_path / f"{ENROLLMENT_ONLY}.seal.json").exists()


def test_sealer_rejects_k5_with_declared_k20_reachable_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        PredictorPackageError,
        match="reachable support pool must equal the declared K",
    ):
        _package(
            tmp_path,
            monkeypatch,
            profile=ENROLLMENT_ONLY,
            k_shot=5,
            declared_support_pool_max_k=20,
        )


@pytest.mark.parametrize(
    ("k_shot", "reachable_per_class"),
    [(5, 6), (10, 20)],
)
def test_sealer_rejects_reachable_support_surplus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    k_shot: int,
    reachable_per_class: int,
) -> None:
    with pytest.raises(
        PredictorPackageError,
        match="exactly K ordered samples",
    ):
        _package(
            tmp_path,
            monkeypatch,
            profile=ENROLLMENT_ONLY,
            k_shot=k_shot,
            support_rows_per_class=reachable_per_class,
        )
    assert not (tmp_path / f"{ENROLLMENT_ONLY}.seal.json").exists()


@pytest.mark.parametrize("k_shot", [1, 5, 10, 20])
def test_loader_materializes_exact_k_support_per_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    k_shot: int,
) -> None:
    class_count = 2
    root, seal, digest, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        class_count=class_count,
        k_shot=k_shot,
    )
    payloads, loaded_manifest, _audit = load_verified_somph_predictor_bundle(
        root,
        detached_seal_path=seal,
        expected_seal_sha256=digest,
    )
    assert manifest["support_pool_max_k"] == k_shot
    assert loaded_manifest["support_pool_max_k"] == k_shot
    for arrays in payloads.values():
        assert arrays["support_tokens"].shape == (class_count * k_shot,)


def test_exact_npz_member_constants_have_no_query_truth_surface() -> None:
    assert SUPPORT_NPZ_MEMBERS == (
        "support_leo_weak_iq",
        "support_class_indices",
        "support_rank_within_class",
        "support_tokens",
        "support_overlay_tokens",
        "support_satellite_seeds",
        "support_post_channel_iq_sha256",
        "manifest_json",
    )
    assert QUERY_NPZ_MEMBERS == (
        "query_leo_weak_iq",
        "query_tokens",
        "query_overlay_tokens",
        "query_satellite_seeds",
        "query_post_channel_iq_sha256",
        "manifest_json",
    )


def _control_root_manifest(*, profile: str, k_shot: int) -> dict:
    class_count = 2
    return {
        "profile": profile,
        "registration_state": "before",
        "registered_class_count": class_count,
        "registered_classes": [
            {"class_index": index, "class_handle": _token("cls_", index + 1)}
            for index in range(class_count)
        ],
        "k_shot": k_shot,
    }


def _validated_control_provenance(
    *, profile: str, rows_by_scenario: dict[str, int]
) -> dict[str, dict[str, dict]]:
    prefix = "sid_" if profile == ENROLLMENT_ONLY else "qid_"
    samples = []
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        for row_index in range(rows_by_scenario[scenario]):
            unique_index = scenario_index * 1000 + row_index + 1
            samples.append(
                {
                    "sample_token": _token(prefix, unique_index),
                    "scenario": scenario,
                    "overlay_token": _token("oid_", 10000 + unique_index),
                    "satellite_seed": 20000 + unique_index,
                    "post_channel_iq_sha256": f"{unique_index:064x}",
                    "source_leo_cache_sha256": "a" * 64,
                    "source_leo_provenance_sha256": "b" * 64,
                }
            )
    return somph_bundle._validate_provenance(
        {
            "schema": SOMPH_OVERLAY_PROVENANCE_SCHEMA,
            "profile": profile,
            "receiver": "20-1",
            "seed": 713101,
            "samples": samples,
        },
        profile=profile,
        receiver="20-1",
        seed=713101,
    )


def test_apply_control_roots_use_actual_nonempty_rows_not_k_shot() -> None:
    row_counts = dict(zip(FORMAL_LEO_WEAK_SCENARIOS, (1, 3, 7)))
    provenance = _validated_control_provenance(
        profile=APPLY_ONLY,
        rows_by_scenario=row_counts,
    )
    roots = [
        somph_bundle._package_control_roots(
            _control_root_manifest(profile=APPLY_ONLY, k_shot=k_shot),
            provenance,
            new_tx_ids=[],
        )
        for k_shot in (1, 5, 10, 20)
    ]
    assert roots[1:] == [roots[0], roots[0], roots[0]]
    assert [len(provenance[scenario]) for scenario in FORMAL_LEO_WEAK_SCENARIOS] == [
        1,
        3,
        7,
    ]

    missing_scene_rows = dict(provenance)
    missing_scene_rows[FORMAL_LEO_WEAK_SCENARIOS[-1]] = {}
    with pytest.raises(PredictorPackageError, match="non-empty scenario"):
        somph_bundle._package_control_roots(
            _control_root_manifest(profile=APPLY_ONLY, k_shot=5),
            missing_scene_rows,
            new_tx_ids=[],
        )


@pytest.mark.parametrize("row_delta", [-1, 1])
def test_enrollment_control_roots_reject_one_row_short_or_long(
    row_delta: int,
) -> None:
    expected_rows = 2 * 5
    provenance = _validated_control_provenance(
        profile=ENROLLMENT_ONLY,
        rows_by_scenario={
            scenario: expected_rows + row_delta
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
    )
    with pytest.raises(PredictorPackageError, match="enrollment.*exact-K"):
        somph_bundle._package_control_roots(
            _control_root_manifest(profile=ENROLLMENT_ONLY, k_shot=5),
            provenance,
            new_tx_ids=[],
        )


def test_package_control_roots_never_opens_or_materializes_iq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = _validated_control_provenance(
        profile=APPLY_ONLY,
        rows_by_scenario={
            scenario: index + 1
            for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
        },
    )

    def forbidden_iq_open(*_args, **_kwargs):
        raise AssertionError("_package_control_roots must not open IQ")

    monkeypatch.setattr(np, "load", forbidden_iq_open)
    monkeypatch.setattr(Path, "open", forbidden_iq_open)
    roots = somph_bundle._package_control_roots(
        _control_root_manifest(profile=APPLY_ONLY, k_shot=20),
        provenance,
        new_tx_ids=[],
    )
    assert roots["package_sample_assignment_sha256"]


def _path_free_authority_roots(*, receiver: str = "20-1") -> dict:
    old_tx_ids = list(somph_bundle.OLD_TX_IDS)
    new_tx_ids = list(somph_bundle.NEW_TX_IDS[:5])
    return {
        "authority_commit_sha256": "8" * 64,
        "authority_lock_sha256": "7" * 64,
        "authority_attestation_sha256": "9" * 64,
        "receiver": receiver,
        "seed": 713101,
        "cache_scope": "stage2_registered",
        "old_tx_ids": old_tx_ids,
        "new_tx_ids": new_tx_ids,
        "cache_sha256_by_scenario": {
            scenario: "a" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "channel_config_sha256_by_scenario": {
            scenario: "e" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "structural_receipt_sha256": "b" * 64,
        "physical_sample_scenario_assignment_policy": (
            somph_bundle.runtime_trust.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": {
            scenario: "1" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "physical_sample_scenario_assignment_sha256": "4" * 64,
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "post_channel_iq_sha256_root_by_scenario": {
            scenario: "2" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "overlay_ids_sha256_by_scenario": {
            scenario: "3" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_role_inputs_root_sha256": "5" * 64,
        "dataset_authority_root_sha256": "6" * 64,
        **somph_bundle.runtime_trust.PHASE2_SINGLE_OBSERVATION_CONTRACT,
    }


def _formal_policy_authorization(
    *,
    manifest: dict,
    seal: dict,
    expected_seal_sha256: str,
    authority_roots: dict,
    policy_sha256: str,
    code_closure_sha256: str,
    package_control_roots: dict,
) -> dict:
    return {
        "schema": somph_bundle.SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
        "status": somph_bundle.SOMPH_FORMAL_POLICY_AUTHORIZATION_STATUS,
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": False,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_detached_seal_sha256": expected_seal_sha256,
        "artifact_member_allowlist_sha256": seal[
            "artifact_member_allowlist_sha256"
        ],
        "manifest_sha256": seal["manifest_sha256"],
        "overlay_provenance_sha256": manifest["overlay_provenance_sha256"],
        "authority_commit_sha256": authority_roots[
            "authority_commit_sha256"
        ],
        "authority_lock_sha256": authority_roots["authority_lock_sha256"],
        "authority_attestation_sha256": authority_roots[
            "authority_attestation_sha256"
        ],
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "stage": manifest["stage"],
        "registration_state": manifest["registration_state"],
        "k_shot": manifest["k_shot"],
        "cache_scope": "stage2_registered",
        "old_tx_ids": authority_roots["old_tx_ids"],
        "new_tx_ids": authority_roots["new_tx_ids"],
        "cache_sha256_by_scenario": authority_roots[
            "cache_sha256_by_scenario"
        ],
        "channel_config_sha256_by_scenario": authority_roots[
            "channel_config_sha256_by_scenario"
        ],
        "structural_receipt_sha256": authority_roots[
            "structural_receipt_sha256"
        ],
        "dataset_authority_root_sha256": authority_roots[
            "dataset_authority_root_sha256"
        ],
        "cache_role_inputs_root_sha256": authority_roots[
            "cache_role_inputs_root_sha256"
        ],
        "physical_sample_ids_sha256_by_scenario": authority_roots[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": authority_roots[
            "physical_sample_scenario_assignment_sha256"
        ],
        "post_channel_iq_sha256_root_by_scenario": authority_roots[
            "post_channel_iq_sha256_root_by_scenario"
        ],
        "overlay_ids_sha256_by_scenario": authority_roots[
            "overlay_ids_sha256_by_scenario"
        ],
        "preflight_code_sha256": sha256_file(Path(somph_bundle.__file__)),
        "formal_policy_sha256": policy_sha256,
        "code_closure_sha256": code_closure_sha256,
        "physical_sample_scenario_assignment_policy": authority_roots[
            "physical_sample_scenario_assignment_policy"
        ],
        "cross_scenario_physical_disjointness_audit": authority_roots[
            "cross_scenario_physical_disjointness_audit"
        ],
        "single_observation_contract_audit": authority_roots[
            "single_observation_contract_audit"
        ],
        "selected_physical_sample_sha256_by_scenario": {
            scenario: f"{index + 6:x}" * 64
            for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
        },
        "selected_overlay_sha256_by_scenario": {
            scenario: f"{index + 9:x}" * 64
            for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
        },
        "selected_membership_assignment_sha256": "c" * 64,
        "support_query_disjointness_status": (
            somph_bundle.SUPPORT_QUERY_DISJOINTNESS_STATUS
        ),
        **{
            key: authority_roots[key]
            for key in somph_bundle.runtime_trust.PHASE2_SINGLE_OBSERVATION_CONTRACT
        },
        **package_control_roots,
    }


def _formal_policy() -> dict:
    return {
        "schema": somph_bundle.SOMPH_FORMAL_POLICY_SCHEMA,
        "status": somph_bundle.SOMPH_FORMAL_POLICY_STATUS,
        "formal_receivers": list(somph_bundle.FORMAL_RECEIVERS),
        "old_tx_ids": list(somph_bundle.OLD_TX_IDS),
        "nested_new_tx_ids": [
            list(somph_bundle.NEW_TX_IDS[:count])
            for count in somph_bundle.FORMAL_NEW_CLASS_COUNTS
        ],
        "cache_scope": "stage2_registered",
        "physical_sample_scenario_assignment_policy": (
            somph_bundle.runtime_trust.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "single_observation_contract": (
            somph_bundle.runtime_trust.PHASE2_SINGLE_OBSERVATION_CONTRACT
        ),
        "required_code_closure_members": list(
            somph_bundle.CODE_CLOSURE_LOGICAL_MEMBERS
        ),
    }


def _test_policy_key_material() -> tuple[bytes, int, bytes]:
    seed = b"\x23" * 32
    hashed = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(
        bytes([hashed[0] & 248])
        + hashed[1:31]
        + bytes([(hashed[31] & 63) | 64]),
        "little",
    )
    public_key = somph_bundle.runtime_trust._ed_encode(
        somph_bundle.runtime_trust._ed_scalar_mult(
            somph_bundle.runtime_trust._ED_B, scalar
        )
    )
    return hashed, scalar, public_key


def _signed_policy_envelope(payload: dict) -> tuple[dict, bytes]:
    hashed, scalar, public_key = _test_policy_key_material()
    envelope = {
        "schema": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA,
        "domain": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN,
        "issuer": somph_bundle.runtime_trust.PINNED_AUTHORITY_ISSUER,
        "key_id": somph_bundle.runtime_trust.PINNED_AUTHORITY_KEY_ID,
        **payload,
        "signature_ed25519_hex": "",
    }
    message = somph_bundle._policy_signature_message(envelope)
    nonce = int.from_bytes(
        hashlib.sha512(hashed[32:] + message).digest(), "little"
    ) % somph_bundle.runtime_trust._ED_L
    encoded_r = somph_bundle.runtime_trust._ed_encode(
        somph_bundle.runtime_trust._ed_scalar_mult(
            somph_bundle.runtime_trust._ED_B, nonce
        )
    )
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public_key + message).digest(), "little"
    ) % somph_bundle.runtime_trust._ED_L
    signature_scalar = (
        nonce + challenge * scalar
    ) % somph_bundle.runtime_trust._ED_L
    envelope["signature_ed25519_hex"] = (
        encoded_r + signature_scalar.to_bytes(32, "little")
    ).hex()
    return envelope, public_key


def _synthetic_policy_verifier(public_key: bytes):
    expected_key_sha256 = hashlib.sha256(public_key).hexdigest()

    def verify(envelope: dict, expected: dict) -> None:
        if set(envelope) != somph_bundle.SIGNED_POLICY_ENVELOPE_KEYS:
            raise PredictorPackageError("synthetic policy envelope schema drift")
        pinned = {
            "schema": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA,
            "domain": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN,
            "issuer": somph_bundle.runtime_trust.PINNED_AUTHORITY_ISSUER,
            "key_id": somph_bundle.runtime_trust.PINNED_AUTHORITY_KEY_ID,
            **expected,
        }
        if any(envelope.get(key) != value for key, value in pinned.items()):
            raise PredictorPackageError("synthetic policy envelope binding drift")
        if hashlib.sha256(public_key).hexdigest() != expected_key_sha256:
            raise PredictorPackageError("synthetic public key drift")
        try:
            signature = bytes.fromhex(str(envelope["signature_ed25519_hex"]))
            somph_bundle.runtime_trust.verify_ed25519(
                public_key,
                somph_bundle._policy_signature_message(envelope),
                signature,
            )
        except (ValueError, KeyError) as exc:
            raise PredictorPackageError("synthetic policy signature invalid") from exc

    return verify


def _policy_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    seal_path: Path,
    seal_sha: str,
    manifest: dict,
    authority_roots: dict,
) -> dict:
    policy_path = tmp_path / "formal_policy.json"
    _write_json(policy_path, _formal_policy())
    policy_sha = sha256_file(policy_path)
    _members, code_closure_sha = somph_bundle._code_closure()
    provenance_payload = json.loads(
        (root / "overlay_provenance.json").read_text(encoding="utf-8")
    )
    provenance = somph_bundle._validate_provenance(
        provenance_payload,
        profile=manifest["profile"],
        receiver=manifest["receiver"],
        seed=manifest["seed"],
    )
    control_roots = somph_bundle._package_control_roots(
        manifest, provenance, new_tx_ids=authority_roots["new_tx_ids"]
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    authorization = _formal_policy_authorization(
        manifest=manifest,
        seal=seal,
        expected_seal_sha256=seal_sha,
        authority_roots=authority_roots,
        policy_sha256=policy_sha,
        code_closure_sha256=code_closure_sha,
        package_control_roots=control_roots,
    )
    authorization_path = tmp_path / "formal_policy_authorization.json"
    _write_json(authorization_path, authorization)
    envelope, public_key = _signed_policy_envelope(
        {
            "authorization_canonical_sha256": somph_bundle.sha256_bytes(
                somph_bundle.canonical_json_bytes(authorization)
            ),
            "formal_policy_sha256": policy_sha,
            "package_root_sha256": manifest["package_root_sha256"],
            "package_detached_seal_sha256": seal_sha,
            "authority_commit_sha256": authority_roots[
                "authority_commit_sha256"
            ],
            "receiver": manifest["receiver"],
            "seed": manifest["seed"],
            "stage": manifest["stage"],
            "registration_state": manifest["registration_state"],
            "k_shot": manifest["k_shot"],
            "code_closure_sha256": code_closure_sha,
        }
    )
    envelope_path = tmp_path / "signed_policy_envelope.json"
    _write_json(envelope_path, envelope)
    synthetic_verifier = _synthetic_policy_verifier(public_key)

    def synthetic_preflight(package_root: str | Path, **kwargs):
        return somph_bundle._preflight_somph_predictor_bundle_with_authority_impl(
            package_root,
            _pinned_policy_verifier=synthetic_verifier,
            **kwargs,
        )

    monkeypatch.setattr(
        somph_bundle,
        "preflight_somph_predictor_bundle_with_authority",
        synthetic_preflight,
    )
    return {
        "formal_policy_path": policy_path,
        "formal_policy_authorization_path": authorization_path,
        "signed_policy_authorization_envelope_path": envelope_path,
        "expected_signed_policy_authorization_envelope_sha256": sha256_file(
            envelope_path
        ),
    }


def test_authority_preflight_passes_without_opening_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
    )
    authority_roots = _path_free_authority_roots()
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=authority_roots,
    )
    assert not hasattr(somph_bundle, "lineage_authority")
    assert not hasattr(somph_bundle, "leo_weak_cache")
    iq_open_calls = {"count": 0}

    def forbidden_iq_open(*_args, **_kwargs):
        iq_open_calls["count"] += 1
        raise AssertionError("authority preflight opened IQ")

    monkeypatch.setattr(somph_bundle, "_inspect_iq_member", forbidden_iq_open)
    monkeypatch.setattr(
        somph_bundle.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("authority preflight called np.load"),
    )
    _manifest, _seal, audit = (
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **policy_inputs,
        )
    )
    assert audit["status"] == "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED"
    assert audit["formal_launch_authority"] is False
    assert audit["formal_metric_claim_allowed"] is False
    assert audit["iq_open_authorized"] is True
    assert audit["iq_archive_opened"] is False
    assert audit["np_load_invoked"] is False
    assert audit["iq_payload_materialized"] is False
    assert audit["signed_path_free_runtime_authorization_verified"] is True
    assert audit["phase2_clean_dataset_reachable"] is False
    assert audit["phase2_clean_cache_reachable"] is False
    assert audit["phase2_clean_control_flow_reachable"] is False
    authorization = json.loads(
        Path(policy_inputs["formal_policy_authorization_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert authorization["formal_launch_authority"] is True
    assert authorization["formal_metric_claim_allowed"] is False
    assert (
        authorization["support_query_disjointness_status"]
        == somph_bundle.SUPPORT_QUERY_DISJOINTNESS_STATUS
    )
    for field in (
        "selected_physical_sample_sha256_by_scenario",
        "selected_overlay_sha256_by_scenario",
        "selected_membership_assignment_sha256",
        "support_query_disjointness_status",
    ):
        assert audit[field] == authorization[field]
    assert "package_physical_sample_ids_sha256_by_scenario" not in authorization
    assert "package_physical_sample_ids_sha256_by_scenario" not in audit
    assert "package_support_token_sha256_by_scenario" in authorization
    assert iq_open_calls["count"] == 0
    assert not any(
        item["relative_path"].endswith(".npz")
        for item in audit["opened_members"]
    )


def test_runtime_authority_api_v1_arguments_fail_closed() -> None:
    signature = inspect.signature(
        somph_bundle.preflight_somph_predictor_bundle_with_authority
    )
    assert "authority_bundle_root" not in signature.parameters
    assert "expected_authority_commit_sha256" not in signature.parameters
    with pytest.raises(TypeError):
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            "unused",
            detached_seal_path="unused",
            expected_seal_sha256="0" * 64,
            authority_bundle_root="forbidden",
            expected_authority_commit_sha256="1" * 64,
            formal_policy_path="unused",
            formal_policy_authorization_path="unused",
            signed_policy_authorization_envelope_path="unused",
            expected_signed_policy_authorization_envelope_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    "drift",
    (
        "missing_selected_physical",
        "selected_physical_scene_missing",
        "selected_overlay_not_sha",
        "selected_assignment_not_sha",
        "metric_claim_true",
        "query_disjointness_claim",
        "legacy_package_physical_key",
    ),
)
def test_support_only_authorization_exact_membership_schema_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
    )
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=_path_free_authority_roots(),
    )
    path = Path(policy_inputs["formal_policy_authorization_path"])
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if drift == "missing_selected_physical":
        del authorization["selected_physical_sample_sha256_by_scenario"]
    elif drift == "selected_physical_scene_missing":
        del authorization["selected_physical_sample_sha256_by_scenario"][
            FORMAL_LEO_WEAK_SCENARIOS[-1]
        ]
    elif drift == "selected_overlay_not_sha":
        authorization["selected_overlay_sha256_by_scenario"][
            FORMAL_LEO_WEAK_SCENARIOS[0]
        ] = "not-a-sha"
    elif drift == "selected_assignment_not_sha":
        authorization["selected_membership_assignment_sha256"] = "not-a-sha"
    elif drift == "metric_claim_true":
        authorization["formal_metric_claim_allowed"] = True
    elif drift == "query_disjointness_claim":
        authorization["support_query_disjointness_status"] = "PASS"
    else:
        authorization["package_physical_sample_ids_sha256_by_scenario"] = {
            scenario: "d" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        }
    with pytest.raises(PredictorPackageError):
        somph_bundle._validate_path_free_authorization_shape(authorization)


def test_runtime_code_closure_excludes_offline_and_cache_modules() -> None:
    assert somph_bundle.CODE_CLOSURE_LOGICAL_MEMBERS == (
        "somph_predictor_bundle.py",
        "somph_runtime_trust.py",
        "stage2_predictor_bundle.py",
    )
    assert not hasattr(somph_bundle, "lineage_authority")
    assert not hasattr(somph_bundle, "leo_weak_cache")
    assert not hasattr(somph_bundle, "formal_matrix")
    tree = ast.parse(inspect.getsource(somph_bundle.runtime_trust))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported == {"hashlib"}
    assert imported_from == {"__future__"}


@pytest.mark.parametrize(
    "name",
    [
        "preflight_somph_predictor_bundle",
        "load_verified_somph_predictor_bundle",
        "load_verified_somph_head_capsule",
    ],
)
def test_formal_runtime_unsigned_reader_surfaces_fail_closed(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(
        somph_bundle,
        "_preflight",
        lambda *_args, **_kwargs: pytest.fail("unsigned reader reached package open"),
    )
    with pytest.raises(PredictorPackageError, match="diagnostic_bundle_loader"):
        getattr(somph_bundle, name)(
            "unused",
            detached_seal_path="unused",
            expected_seal_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("validator", "payload"),
    [
        (
            somph_bundle._validate_formal_policy,
            {**_formal_policy(), "cache_scope": "ManySig.pkl"},
        ),
        (
            somph_bundle._validate_path_free_authorization_shape,
            {
                "schema": somph_bundle.SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
                "build_spec": "forbidden",
            },
        ),
    ],
)
def test_runtime_policy_inputs_reject_path_raw_build_loader_reachability(
    validator, payload: dict
) -> None:
    with pytest.raises(PredictorPackageError, match="forbidden reachability|forbidden path"):
        validator(payload)


@pytest.mark.parametrize("legacy_surface", ["authorization", "envelope"])
def test_signed_policy_v1_surfaces_fail_before_iq(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_surface: str,
) -> None:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
    )
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=_path_free_authority_roots(),
    )
    if legacy_surface == "authorization":
        path = policy_inputs["formal_policy_authorization_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema"] = "cvs.phase2.somph_formal_row_policy_authorization.v1"
        _write_json(path, payload)
    else:
        path = policy_inputs["signed_policy_authorization_envelope_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema"] = (
            "cvs.phase2.somph_signed_policy_authorization_envelope.v1"
        )
        payload["domain"] = "cvs.somph.formal_policy_authorization.ed25519.v1"
        _write_json(path, payload)
        policy_inputs["expected_signed_policy_authorization_envelope_sha256"] = (
            sha256_file(path)
        )
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *_args, **_kwargs: pytest.fail("legacy v1 failure opened IQ"),
    )
    monkeypatch.setattr(
        somph_bundle.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("legacy v1 failure called np.load"),
    )
    with pytest.raises(PredictorPackageError):
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **policy_inputs,
        )


@pytest.mark.parametrize(
    "forbidden_payload",
    [
        {"nested": {"build_spec": {"source": "opaque"}}},
        {"nested": {"raw_member": "E:/sealed/ManySig.pkl"}},
    ],
)
def test_runtime_authorization_recursive_reachability_guard(
    forbidden_payload: dict,
) -> None:
    with pytest.raises(PredictorPackageError, match="forbidden reachability"):
        somph_bundle._reject_runtime_authorization_reachability(
            forbidden_payload
        )


def test_authority_preflight_rejects_d8b_without_opening_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
        receiver="1-20",
    )
    authority_roots = _path_free_authority_roots(receiver="1-20")
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=authority_roots,
    )
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *_args, **_kwargs: pytest.fail("D8b failure opened IQ"),
    )
    monkeypatch.setattr(
        somph_bundle.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("D8b failure called np.load"),
    )
    with pytest.raises(PredictorPackageError, match="formal receiver"):
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **policy_inputs,
        )


def test_authority_preflight_missing_signed_root_fails_before_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
    )
    authority_roots = _path_free_authority_roots()
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=authority_roots,
    )
    authorization_path = policy_inputs["formal_policy_authorization_path"]
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    del authorization["dataset_authority_root_sha256"]
    _write_json(authorization_path, authorization)
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *_args, **_kwargs: pytest.fail("missing authority opened IQ"),
    )
    monkeypatch.setattr(
        somph_bundle.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("missing authority called np.load"),
    )
    with pytest.raises(PredictorPackageError):
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **policy_inputs,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "signature",
        "actual_policy",
        "authorization_root",
        "selected_membership",
        "code_closure",
    ],
)
def test_signed_policy_binding_drift_fails_before_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
    )
    authority_roots = _path_free_authority_roots()
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=authority_roots,
    )
    if tamper == "signature":
        path = policy_inputs["signed_policy_authorization_envelope_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["signature_ed25519_hex"] = "00" * 64
        _write_json(path, payload)
        policy_inputs["expected_signed_policy_authorization_envelope_sha256"] = (
            sha256_file(path)
        )
    elif tamper == "actual_policy":
        path = policy_inputs["formal_policy_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "TAMPERED"
        _write_json(path, payload)
    elif tamper == "authorization_root":
        path = policy_inputs["formal_policy_authorization_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["package_sample_assignment_sha256"] = "0" * 64
        _write_json(path, payload)
    elif tamper == "selected_membership":
        path = policy_inputs["formal_policy_authorization_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selected_membership_assignment_sha256"] = "0" * 64
        _write_json(path, payload)
    else:
        members, _closure = somph_bundle._code_closure()
        monkeypatch.setattr(
            somph_bundle,
            "_code_closure",
            lambda: (members, "0" * 64),
        )
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *_args, **_kwargs: pytest.fail("binding drift opened IQ"),
    )
    monkeypatch.setattr(
        somph_bundle.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("binding drift called np.load"),
    )
    with pytest.raises(PredictorPackageError):
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **policy_inputs,
        )


def test_signed_selected_physical_root_cannot_alias_opaque_support_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
    )
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=_path_free_authority_roots(),
    )
    authorization_path = Path(
        policy_inputs["formal_policy_authorization_path"]
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    authorization["selected_physical_sample_sha256_by_scenario"][scenario] = (
        authorization["package_support_token_sha256_by_scenario"][scenario]
    )
    _write_json(authorization_path, authorization)

    envelope_path = Path(
        policy_inputs["signed_policy_authorization_envelope_path"]
    )
    old_envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    payload_keys = somph_bundle.SIGNED_POLICY_ENVELOPE_KEYS - {
        "schema",
        "domain",
        "issuer",
        "key_id",
        "signature_ed25519_hex",
    }
    envelope_payload = {key: old_envelope[key] for key in payload_keys}
    envelope_payload["authorization_canonical_sha256"] = somph_bundle.sha256_bytes(
        somph_bundle.canonical_json_bytes(authorization)
    )
    envelope, _public_key = _signed_policy_envelope(envelope_payload)
    _write_json(envelope_path, envelope)
    policy_inputs["expected_signed_policy_authorization_envelope_sha256"] = (
        sha256_file(envelope_path)
    )
    monkeypatch.setattr(
        somph_bundle,
        "_inspect_iq_member",
        lambda *_args, **_kwargs: pytest.fail("opaque root alias opened IQ"),
    )
    monkeypatch.setattr(
        somph_bundle.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("opaque root alias called np.load"),
    )
    with pytest.raises(PredictorPackageError, match="opaque tokens"):
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **policy_inputs,
        )


def _preauthorized_enrollment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, dict, dict, dict]:
    root, seal_path, seal_sha, manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        stage="stage2b",
        registration_state="before",
        class_count=6,
        k_shot=10,
    )
    authority_roots = _path_free_authority_roots()
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        authority_roots=authority_roots,
    )
    _manifest, seal, preflight = (
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **policy_inputs,
        )
    )
    return root, seal_path, seal_sha, manifest, seal, {
        "preflight": preflight,
        "policy_inputs": policy_inputs,
    }


def test_ordinary_payload_and_self_constructed_receipt_cannot_promote() -> None:
    with pytest.raises(PredictorPackageError, match="token-sealed"):
        somph_bundle.finalize_somph_enrollment_authority_after_materialization(
            {"materialized_payloads": {}, "materialization_receipt": {}}
        )
    with pytest.raises(TypeError):
        somph_bundle.finalize_somph_enrollment_authority_after_materialization(
            manifest={},
            seal={},
            authority_preflight_audit={},
            materialized_payloads={},
            materialization_receipt={},
        )


def test_legacy_audit_handoff_materializer_fails_before_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        somph_bundle,
        "_materialize_iq",
        lambda *_args, **_kwargs: pytest.fail("legacy audit handoff opened IQ"),
    )
    with pytest.raises(PredictorPackageError, match="atomic entry"):
        somph_bundle.materialize_somph_enrollment_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            authority_preflight_audit=context["preflight"],
        )


def test_synthetic_key_cannot_open_atomic_production_iq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        somph_bundle,
        "_materialize_iq",
        lambda *_args, **_kwargs: pytest.fail("synthetic key opened production IQ"),
    )
    with pytest.raises(PredictorPackageError, match="signed policy authorization"):
        somph_bundle.materialize_somph_enrollment_with_signed_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            **context["policy_inputs"],
        )


def test_production_module_exports_no_capability_issuer_or_test_factory() -> None:
    forbidden = (
        "SomphAuthorityPreflightEvidence",
        "_issue_somph_authority_preflight_evidence",
        "_consume_somph_authority_preflight_evidence",
        "_make_authority_preflight_capability_api",
        "_make_test_authority_preflight",
        "_make_test_signed_policy_envelope_verifier",
    )
    assert all(not hasattr(somph_bundle, name) for name in forbidden)
    signature = inspect.signature(
        somph_bundle.materialize_somph_enrollment_with_signed_authority
    )
    assert not (
        {"authority_preflight_audit", "capability", "public_key", "verifier"}
        & set(signature.parameters)
    )


def test_package_root_reached_through_parent_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_parent = tmp_path / "real"
    root, seal_path, seal_sha, _manifest = _package(
        real_parent,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
    )
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PredictorPackageError, match="parent symlink/reparse"):
        preflight_somph_predictor_bundle(
            alias / root.name,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
        )


def test_package_root_parent_reparse_marker_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "parent" / "enrollment_only"
    root.mkdir(parents=True)
    real = somph_bundle._is_reparse_or_symlink

    def marked(path: Path) -> bool:
        return path == root.parent or real(path)

    monkeypatch.setattr(somph_bundle, "_is_reparse_or_symlink", marked)
    with pytest.raises(PredictorPackageError, match="parent symlink/reparse"):
        somph_bundle._ensure_root(root)
