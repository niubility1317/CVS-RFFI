from __future__ import annotations

import json
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
    registration_state: str,
    sample_token_offset: int = 0,
    swap_first_tokens: bool = False,
) -> list[dict]:
    count = class_count * 20
    iq = np.arange(count * 2 * 4, dtype=np.float32).reshape(count, 2, 4)
    tokens = [
        _token("sid_", sample_token_offset + index + 1) for index in range(count)
    ]
    if swap_first_tokens:
        tokens[0], tokens[1] = tokens[1], tokens[0]
    scenario_offset = FORMAL_LEO_WEAK_SCENARIOS.index(scenario) * 1000
    overlays = [
        _token("oid_", 10000 + scenario_offset + index) for index in range(count)
    ]
    seeds = np.arange(
        20000 + scenario_offset, 20000 + scenario_offset + count, dtype=np.int64
    )
    hashes = [iq_row_sha256(row) for row in iq]
    labels = np.repeat(np.arange(class_count, dtype=np.int64), 20)
    ranks = np.tile(np.arange(20, dtype=np.int64), class_count)
    embedded = {
        "schema": SOMPH_SUPPORT_IQ_SCHEMA,
        "scenario": scenario,
        "registration_state": registration_state,
        "registered_class_count": class_count,
        "support_pool_max_k": 20,
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
    row_handle: str = "row_" + "4" * 64,
    row_manifest_sha256: str = "5" * 64,
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
            "receiver": "20-1",
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
        sample_token_offset = (
            1000
            if drift_last_scenario_tokens
            and scenario == FORMAL_LEO_WEAK_SCENARIOS[-1]
            else 0
        )
        if profile == ENROLLMENT_ONLY:
            samples.extend(
                _support_npz(
                    root / f"support_{scenario}.npz",
                    scenario=scenario,
                    class_count=class_count,
                    registration_state=registration_state,
                    sample_token_offset=sample_token_offset,
                    swap_first_tokens=(
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
            "receiver": "20-1",
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
        receiver="20-1",
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
def test_cross_scenario_physical_sample_token_set_drift_is_rejected(
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
        match="physical sample-token set drifts across LEO_weak scenarios",
    ):
        load_verified_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_cross_scenario_support_token_assignment_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, seal, digest, _manifest = _package(
        tmp_path,
        monkeypatch,
        profile=ENROLLMENT_ONLY,
        drift_last_support_assignment=True,
    )
    with pytest.raises(
        PredictorPackageError,
        match="support token/class/rank mapping drifts",
    ):
        load_verified_somph_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


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
