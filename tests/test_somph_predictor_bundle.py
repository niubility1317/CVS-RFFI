from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.somph_predictor_bundle as somph_bundle
import cvsrffi.somph_predictor_runtime as somph_runtime
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
    load_verified_somph_predictor_bundle,
    preflight_somph_predictor_bundle,
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
    assert audit["status"] == "STRUCTURAL_SELF_CONSISTENCY_PASS"
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
    root, seal, digest, _manifest = _package(
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


def _formal_authority_payloads(*, receiver: str = "20-1") -> tuple[dict, dict, dict]:
    old_tx_ids = list(somph_bundle.OLD_TX_IDS)
    new_tx_ids = list(somph_bundle.NEW_TX_IDS[:5])
    lock = {
        "receiver": receiver,
        "seed": 713101,
        "cache_scope": "stage2_registered",
        "old_tx_ids": old_tx_ids,
        "new_tx_ids": new_tx_ids,
        "cache_sha256_by_scenario": {
            scenario: "a" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "physical_sample_scenario_assignment_policy": (
            somph_bundle.lineage_authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
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
        "datasets": [
            {
                "role": "target_old",
                "path": "E:/sealed/offline/ManySig.pkl",
                "sha256": "6" * 64,
                "size_bytes": 1,
                "tx_ids": old_tx_ids,
            },
            {
                "role": "target_new",
                "path": "E:/sealed/offline/ManyTx.pkl",
                "sha256": "7" * 64,
                "size_bytes": 1,
                "tx_ids": new_tx_ids,
            },
        ],
        **somph_bundle.lineage_authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
    }
    attestation = {
        "structural_receipt_sha256": "b" * 64,
        "dataset_authority_root_sha256": "6" * 64,
    }
    commit = {
        "authority_lock_sha256": "7" * 64,
        "members": [
            {
                "name": somph_bundle.lineage_authority.AUTHORITY_ATTESTATION_NAME,
                "sha256": "9" * 64,
            }
        ],
    }
    return lock, attestation, commit


def _formal_policy_authorization(
    *,
    manifest: dict,
    seal: dict,
    expected_seal_sha256: str,
    expected_commit_sha256: str,
    lock: dict,
    attestation: dict,
    commit: dict,
    policy_sha256: str,
    code_closure_sha256: str,
    package_control_roots: dict,
) -> dict:
    return {
        "schema": somph_bundle.SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
        "status": somph_bundle.SOMPH_FORMAL_POLICY_AUTHORIZATION_STATUS,
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": True,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_detached_seal_sha256": expected_seal_sha256,
        "artifact_member_allowlist_sha256": seal[
            "artifact_member_allowlist_sha256"
        ],
        "manifest_sha256": seal["manifest_sha256"],
        "overlay_provenance_sha256": manifest["overlay_provenance_sha256"],
        "authority_commit_sha256": expected_commit_sha256,
        "authority_lock_sha256": commit["authority_lock_sha256"],
        "authority_attestation_sha256": "9" * 64,
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "stage": manifest["stage"],
        "registration_state": manifest["registration_state"],
        "k_shot": manifest["k_shot"],
        "cache_scope": "stage2_registered",
        "old_tx_ids": lock["old_tx_ids"],
        "new_tx_ids": lock["new_tx_ids"],
        "dataset_authority_root_sha256": attestation[
            "dataset_authority_root_sha256"
        ],
        "cache_role_inputs_root_sha256": lock[
            "cache_role_inputs_root_sha256"
        ],
        "physical_sample_ids_sha256_by_scenario": lock[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": lock[
            "physical_sample_scenario_assignment_sha256"
        ],
        "post_channel_iq_sha256_root_by_scenario": lock[
            "post_channel_iq_sha256_root_by_scenario"
        ],
        "overlay_ids_sha256_by_scenario": lock[
            "overlay_ids_sha256_by_scenario"
        ],
        "preflight_code_sha256": sha256_file(Path(somph_bundle.__file__)),
        "formal_policy_sha256": policy_sha256,
        "code_closure_sha256": code_closure_sha256,
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
        "old_dataset_basename": "ManySig.pkl",
        "new_dataset_basename": "ManyTx.pkl",
        "physical_sample_scenario_assignment_policy": (
            somph_bundle.lineage_authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "single_observation_contract": (
            somph_bundle.lineage_authority.PHASE2_SINGLE_OBSERVATION_CONTRACT
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
    public_key = somph_bundle.lineage_authority._ed_encode(
        somph_bundle.lineage_authority._ed_scalar_mult(
            somph_bundle.lineage_authority._ED_B, scalar
        )
    )
    return hashed, scalar, public_key


def _signed_policy_envelope(payload: dict) -> tuple[dict, bytes]:
    hashed, scalar, public_key = _test_policy_key_material()
    envelope = {
        "schema": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA,
        "domain": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN,
        "issuer": somph_bundle.lineage_authority.PINNED_AUTHORITY_ISSUER,
        "key_id": somph_bundle.lineage_authority.PINNED_AUTHORITY_KEY_ID,
        **payload,
        "signature_ed25519_hex": "",
    }
    message = somph_bundle._policy_signature_message(envelope)
    nonce = int.from_bytes(
        hashlib.sha512(hashed[32:] + message).digest(), "little"
    ) % somph_bundle.lineage_authority._ED_L
    encoded_r = somph_bundle.lineage_authority._ed_encode(
        somph_bundle.lineage_authority._ed_scalar_mult(
            somph_bundle.lineage_authority._ED_B, nonce
        )
    )
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public_key + message).digest(), "little"
    ) % somph_bundle.lineage_authority._ED_L
    signature_scalar = (
        nonce + challenge * scalar
    ) % somph_bundle.lineage_authority._ED_L
    envelope["signature_ed25519_hex"] = (
        encoded_r + signature_scalar.to_bytes(32, "little")
    ).hex()
    return envelope, public_key


def _policy_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    seal_path: Path,
    seal_sha: str,
    manifest: dict,
    lock: dict,
    attestation: dict,
    commit: dict,
    expected_commit_sha: str,
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
        manifest, provenance, new_tx_ids=lock["new_tx_ids"]
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    authorization = _formal_policy_authorization(
        manifest=manifest,
        seal=seal,
        expected_seal_sha256=seal_sha,
        expected_commit_sha256=expected_commit_sha,
        lock=lock,
        attestation=attestation,
        commit=commit,
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
            "authority_commit_sha256": expected_commit_sha,
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
    monkeypatch.setattr(
        somph_bundle,
        "preflight_somph_predictor_bundle_with_authority",
        somph_bundle._make_test_authority_preflight(public_key),
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
    lock, attestation, commit = _formal_authority_payloads()
    expected_commit_sha = "8" * 64
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        lock=lock,
        attestation=attestation,
        commit=commit,
        expected_commit_sha=expected_commit_sha,
    )
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "verify_somph_lineage_authority_bundle",
        lambda *_args, **_kwargs: (lock, attestation, commit),
    )
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
            authority_bundle_root=tmp_path / "authority_bundle",
            expected_authority_commit_sha256=expected_commit_sha,
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
    assert iq_open_calls["count"] == 0
    assert not any(
        item["relative_path"].endswith(".npz")
        for item in audit["opened_members"]
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
    lock, attestation, commit = _formal_authority_payloads(receiver="1-20")
    lock["datasets"][0]["path"] = "E:/sealed/offline/ManyTx.pkl"
    expected_commit_sha = "8" * 64
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        lock=lock,
        attestation=attestation,
        commit=commit,
        expected_commit_sha=expected_commit_sha,
    )
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "verify_somph_lineage_authority_bundle",
        lambda *_args, **_kwargs: (lock, attestation, commit),
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
            authority_bundle_root=tmp_path / "authority_bundle",
            expected_authority_commit_sha256=expected_commit_sha,
            **policy_inputs,
        )


def test_authority_preflight_missing_bundle_field_fails_before_iq(
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
    lock, attestation, commit = _formal_authority_payloads()
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        lock=lock,
        attestation=attestation,
        commit=commit,
        expected_commit_sha="8" * 64,
    )
    del attestation["dataset_authority_root_sha256"]
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "verify_somph_lineage_authority_bundle",
        lambda *_args, **_kwargs: (lock, attestation, commit),
    )
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
            authority_bundle_root=tmp_path / "authority_bundle",
            expected_authority_commit_sha256="8" * 64,
            **policy_inputs,
        )


@pytest.mark.parametrize(
    "tamper",
    ["signature", "actual_policy", "authorization_root", "code_closure"],
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
    lock, attestation, commit = _formal_authority_payloads()
    expected_commit_sha = "8" * 64
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        lock=lock,
        attestation=attestation,
        commit=commit,
        expected_commit_sha=expected_commit_sha,
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
    else:
        members, _closure = somph_bundle._code_closure()
        monkeypatch.setattr(
            somph_bundle,
            "_code_closure",
            lambda: (members, "0" * 64),
        )
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "verify_somph_lineage_authority_bundle",
        lambda *_args, **_kwargs: (lock, attestation, commit),
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
            authority_bundle_root=tmp_path / "authority_bundle",
            expected_authority_commit_sha256=expected_commit_sha,
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
    lock, attestation, commit = _formal_authority_payloads()
    expected_commit_sha = "8" * 64
    policy_inputs = _policy_inputs(
        tmp_path,
        monkeypatch,
        root=root,
        seal_path=seal_path,
        seal_sha=seal_sha,
        manifest=manifest,
        lock=lock,
        attestation=attestation,
        commit=commit,
        expected_commit_sha=expected_commit_sha,
    )
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "verify_somph_lineage_authority_bundle",
        lambda *_args, **_kwargs: (lock, attestation, commit),
    )
    _manifest, seal, preflight = (
        somph_bundle.preflight_somph_predictor_bundle_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            authority_bundle_root=tmp_path / "authority_bundle",
            expected_authority_commit_sha256=expected_commit_sha,
            **policy_inputs,
        )
    )
    return root, seal_path, seal_sha, manifest, seal, {
        "preflight": preflight,
        "policy_inputs": policy_inputs,
        "expected_commit_sha": expected_commit_sha,
    }


def test_post_materialization_finalizer_is_the_only_formal_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    preflight = context["preflight"]
    assert preflight["formal_launch_authority"] is False
    evidence = somph_bundle.materialize_somph_enrollment_with_authority(
        root,
        detached_seal_path=seal_path,
        expected_seal_sha256=seal_sha,
        authority_preflight_audit=preflight,
    )
    final = somph_bundle.finalize_somph_enrollment_authority_after_materialization(
        evidence
    )
    assert final["status"] == "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
    assert final["formal_launch_authority"] is True
    assert final["formal_metric_claim_allowed"] is True
    assert final["iq_payload_materialized"] is True
    assert final["verified_materialization_evidence_sha256"] == (
        evidence.evidence_sha256
    )
    with pytest.raises(PredictorPackageError, match="fresh token-sealed"):
        somph_bundle.finalize_somph_enrollment_authority_after_materialization(
            evidence
        )


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


@pytest.mark.parametrize("tamper", ["iq", "token", "overlay", "seed"])
def test_replaced_archive_observation_cannot_issue_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    preflight = context["preflight"]
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    path = root / f"support_{scenario}.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if tamper == "iq":
        arrays["support_leo_weak_iq"][0, 0, 0] += 1
        arrays["support_post_channel_iq_sha256"][0] = iq_row_sha256(
            arrays["support_leo_weak_iq"][0]
        )
    elif tamper == "token":
        arrays["support_tokens"][0] = "sid_" + "f" * 64
    elif tamper == "overlay":
        arrays["support_overlay_tokens"][0] = "oid_" + "f" * 64
    else:
        arrays["support_satellite_seeds"][0] += 1
    path.unlink()
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(PredictorPackageError, match="changed after preflight"):
        somph_bundle.materialize_somph_enrollment_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            authority_preflight_audit=preflight,
        )


def test_forged_materializer_return_is_rejected_by_preauthorized_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    preflight = context["preflight"]
    real_materialize = somph_bundle._materialize_iq

    def forged_materialize(*args, **kwargs):
        arrays, embedded = real_materialize(*args, **kwargs)
        arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
        arrays["support_tokens"][0] = "sid_" + "f" * 64
        return arrays, embedded

    monkeypatch.setattr(somph_bundle, "_materialize_iq", forged_materialize)
    with pytest.raises(PredictorPackageError, match="provenance|preauthorized roots"):
        somph_bundle.materialize_somph_enrollment_with_authority(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            authority_preflight_audit=preflight,
        )


def test_materialized_evidence_payload_arrays_are_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    evidence = somph_bundle.materialize_somph_enrollment_with_authority(
        root,
        detached_seal_path=seal_path,
        expected_seal_sha256=seal_sha,
        authority_preflight_audit=context["preflight"],
    )
    array = evidence.materialized_payloads[FORMAL_LEO_WEAK_SCENARIOS[0]][
        "support_leo_weak_iq"
    ]
    assert array.flags.writeable is False
    with pytest.raises(ValueError):
        array[0, 0, 0] += 1


def test_copied_or_unissued_materialized_evidence_cannot_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    evidence = somph_bundle.materialize_somph_enrollment_with_authority(
        root,
        detached_seal_path=seal_path,
        expected_seal_sha256=seal_sha,
        authority_preflight_audit=context["preflight"],
    )
    copied = copy.copy(evidence)
    with pytest.raises(PredictorPackageError, match="fresh token-sealed"):
        somph_bundle.finalize_somph_enrollment_authority_after_materialization(
            copied
        )
    unissued = object.__new__(somph_bundle.SomphMaterializedEnrollmentEvidence)
    with pytest.raises(PredictorPackageError, match="fresh token-sealed"):
        somph_bundle.finalize_somph_enrollment_authority_after_materialization(
            unissued
        )
    final = somph_bundle.finalize_somph_enrollment_authority_after_materialization(
        evidence
    )
    assert final["formal_launch_authority"] is True


def test_production_policy_verifier_ignores_mutated_trust_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production_preflight = somph_bundle.preflight_somph_predictor_bundle_with_authority
    root, seal_path, seal_sha, _manifest, _seal, context = (
        _preauthorized_enrollment(tmp_path, monkeypatch)
    )
    _hashed, _scalar, test_public_key = _test_policy_key_material()
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "PINNED_AUTHORITY_PUBLIC_KEY_HEX",
        test_public_key.hex(),
    )
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "PINNED_AUTHORITY_PUBLIC_KEY_SHA256",
        hashlib.sha256(test_public_key).hexdigest(),
    )
    with pytest.raises(PredictorPackageError, match="signed policy authorization"):
        production_preflight(
            root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha,
            authority_bundle_root=tmp_path / "authority_bundle",
            expected_authority_commit_sha256=context["expected_commit_sha"],
            **context["policy_inputs"],
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
