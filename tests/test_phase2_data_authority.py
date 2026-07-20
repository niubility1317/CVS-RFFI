from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import phase2_data_authority as authority  # noqa: E402
from cvsrffi.phase2_runtime_contract import (  # noqa: E402
    PHASE2_FULL_CONTRACT,
    PHASE2_SINGLE_OBSERVATION_CONTRACT,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    PREDICTOR_INPUT_STAGE,
    PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
    PREDICTOR_PACKAGE_SEAL_SCHEMA,
    QUERY_NPZ_MEMBERS,
    QUERY_SCHEMA,
    SUPPORT_NPZ_MEMBERS,
    SUPPORT_SCHEMA,
    canonical_json_bytes,
    package_root_sha256,
)


def _sha(tag: str) -> str:
    return authority.sha256_bytes(tag.encode("utf-8"))


def _scenario_map(prefix: str) -> dict[str, str]:
    return {scenario: _sha(f"{prefix}:{scenario}") for scenario in FORMAL_LEO_WEAK_SCENARIOS}


def _member(
    role: str,
    *,
    digest: str,
    scenario: str | None = None,
    schema: str = "test.method.v1",
    npz_members: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "relative_path": role.replace(":", "_") + (".npz" if scenario else ".bin"),
        "sha256": digest,
        "size_bytes": 101,
        "artifact_role": role,
        "schema": schema,
        "scenario": scenario,
        "npz_members": list(npz_members),
    }


def _control_payloads(
    *,
    method_salt: str = "m0",
    data_salt: str = "d0",
    split_salt: str = "s0",
    k_shot: int = 5,
    stage: str = "stage2c",
    seed: int = 713101,
) -> tuple[dict, bytes, dict, dict, bytes, dict]:
    members = [
        _member("checkpoint", digest=_sha(f"{method_salt}:checkpoint")),
        _member("adapter", digest=_sha(f"{method_salt}:adapter")),
        _member("head", digest=_sha(f"{method_salt}:head")),
        _member("tta_policy", digest=_sha(f"{method_salt}:tta")),
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        members.extend(
            [
                _member(
                    f"support:{scenario}",
                    digest=_sha(f"{data_salt}:support-member:{scenario}"),
                    scenario=scenario,
                    schema=SUPPORT_SCHEMA,
                    npz_members=SUPPORT_NPZ_MEMBERS,
                ),
                _member(
                    f"query:{scenario}",
                    digest=_sha(f"{data_salt}:query-member:{scenario}"),
                    scenario=scenario,
                    schema=QUERY_SCHEMA,
                    npz_members=QUERY_NPZ_MEMBERS,
                ),
            ]
        )
    registry = [
        {
            "class_index": index,
            "class_handle": f"cls_{_sha(f'{method_salt}:class:{index}')}",
        }
        for index in range(3)
    ]
    manifest = {
        "schema": PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
        "artifact_stage": PREDICTOR_INPUT_STAGE,
        "stage": stage,
        "receiver": "20-1",
        "seed": seed,
        "new_class_count": 0 if stage == "stage2b" else 1,
        "support_pool_max_k": 20,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "registered_class_count": 3,
        "registered_classes": registry,
        "candidate_lock_sha256": _sha(f"{method_salt}:candidate"),
        "members": members,
        "package_root_sha256": package_root_sha256(members),
        **PHASE2_FULL_CONTRACT,
    }
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    seal = {
        "schema": PREDICTOR_PACKAGE_SEAL_SCHEMA,
        "manifest_relative_path": "package_manifest.json",
        "manifest_sha256": authority.sha256_bytes(manifest_raw),
        "manifest_size_bytes": len(manifest_raw),
        "package_root_sha256": manifest["package_root_sha256"],
        "artifact_member_allowlist_sha256": manifest["package_root_sha256"],
    }
    seal_raw = canonical_json_bytes(seal) + b"\n"

    cache_physical = _scenario_map(f"{data_salt}:cache-physical")
    cache_manifest = {
        "schema": "cvs.leo_weak.cache_set.v3",
        "artifact_stage": "phase2_single_observation_cache",
        "cache_scope": "stage2_registered",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "physical_sample_ids_sha256_by_scenario": cache_physical,
        "physical_sample_scenario_assignment_sha256": _sha(
            f"{data_salt}:cache-assignment"
        ),
        **PHASE2_SINGLE_OBSERVATION_CONTRACT,
    }
    cache_audits = {
        scenario: {
            "path": f"cache_{scenario}.npz",
            "sha256": _sha(f"{data_salt}:cache-file:{scenario}"),
            "schema": "cvs.leo_weak.cache.v3",
            "scenario": scenario,
            "row_count": 100,
            "roles": ["target_new", "target_old"],
            "satellite_seeds": [713101],
            "physical_sample_ids_sha256": cache_physical[scenario],
            "post_channel_iq_sha256_root": _sha(
                f"{data_salt}:cache-iq:{scenario}"
            ),
            "overlay_ids_sha256": _sha(f"{data_salt}:overlay:{scenario}"),
            "manifest_sha256": _sha(f"{data_salt}:manifest:{scenario}"),
            "forbidden_members_checked_before_iq_read": True,
            "clean_sample_access": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "phase2_physical_sample_root_id_policy": (
                "immutable_preoverlay_lineage_token"
            ),
        }
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    cache_audit = {
        "path": "cache_set.json",
        "sha256": _sha(f"{data_salt}:cache-manifest-file"),
        "scope": "stage2_registered",
        "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
        "physical_sample_count": 300,
        "physical_sample_observation_count": 300,
        "physical_sample_count_by_scenario": {
            scenario: 100 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "physical_sample_ids_sha256_by_scenario": cache_physical,
        "physical_sample_scenario_assignment_sha256": cache_manifest[
            "physical_sample_scenario_assignment_sha256"
        ],
        "cache_audits": cache_audits,
        "clean_sample_access": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_physical_sample_root_id_policy": (
            "immutable_preoverlay_lineage_token"
        ),
        "phase2_single_observation_compliant": True,
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
    }
    audit = {
        "schema": authority.OFFLINE_AUDIT_SCHEMA,
        "status": "PASS",
        "target_cache_manifest": cache_manifest,
        "target_cache_audit": cache_audit,
        "predictor_package_root_sha256": manifest["package_root_sha256"],
        "predictor_package_seal_sha256": authority.sha256_bytes(seal_raw),
        "predictor_scorer_roots_distinct": True,
        "opaque_token_secret_persisted": False,
        "same_scenario_support_query_physical_disjointness": "PASS",
        "cross_scenario_selected_physical_disjointness": "PASS",
        "cross_scenario_opaque_token_disjointness": "PASS",
        "registered_class_rank_structure_consistent": "PASS",
    }
    audit_raw = canonical_json_bytes(audit) + b"\n"
    data_descriptors = [
        {field: item[field] for field in authority.DATA_MEMBER_DESCRIPTOR_FIELDS}
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
        for item in members
        if item["artifact_role"] in {f"support:{scenario}", f"query:{scenario}"}
    ]
    # The producer orders support then query within each scenario.
    data_descriptors = []
    by_role = {item["artifact_role"]: item for item in members}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for prefix in ("support", "query"):
            item = by_role[f"{prefix}:{scenario}"]
            data_descriptors.append(
                {field: item[field] for field in authority.DATA_MEMBER_DESCRIPTOR_FIELDS}
            )
    final_registry = [item["class_handle"] for item in registry]
    old_registry = final_registry if stage == "stage2b" else final_registry[:2]
    old_identity_root = _sha(
        "registry:final:tx-a,tx-b,tx-c"
        if stage == "stage2b"
        else "registry:old:tx-a,tx-b"
    )
    final_identity_root = _sha("registry:final:tx-a,tx-b,tx-c")
    commit = {
        "schema": authority.DATA_VALIDATION_COMMIT_SCHEMA,
        "status": authority.DATA_VALIDATION_COMMIT_STATUS,
        "protocol_schema": authority.PROTOCOL_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "predictor_package_root_sha256": manifest["package_root_sha256"],
        "predictor_package_seal_sha256": authority.sha256_bytes(seal_raw),
        "predictor_package_manifest_sha256": authority.sha256_bytes(manifest_raw),
        "offline_build_audit_sha256": authority.sha256_bytes(audit_raw),
        "offline_build_audit_canonical_sha256": authority.canonical_sha256(audit),
        "target_cache_manifest_file_sha256": cache_audit["sha256"],
        "target_cache_manifest_canonical_sha256": authority.canonical_sha256(
            cache_manifest
        ),
        "target_cache_audit_canonical_sha256": authority.canonical_sha256(cache_audit),
        "data_member_descriptors_root_sha256": authority.canonical_sha256(
            data_descriptors
        ),
        "receiver": "20-1",
        "seed": seed,
        "stage": stage,
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "k_shot": k_shot,
        "old_registry": old_registry,
        "final_registry": final_registry,
        "old_registry_identity_root_sha256": old_identity_root,
        "final_registry_identity_root_sha256": final_identity_root,
        "support_count_by_scenario": {
            scenario: 3 * k_shot for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "support_count_by_class_by_scenario": {
            scenario: {class_handle: k_shot for class_handle in final_registry}
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "support_physical_ids_root_by_class_by_scenario": {
            scenario: {
                class_handle: _sha(
                    f"{split_salt}:support-class-physical:{scenario}:{class_handle}"
                )
                for class_handle in final_registry
            }
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "query_count_by_scenario": {
            scenario: 30 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "ordered_support_opaque_token_root_sha256_by_scenario": _scenario_map(
            f"{split_salt}:support-token"
        ),
        "ordered_query_opaque_token_root_sha256_by_scenario": _scenario_map(
            f"{split_salt}:query-token"
        ),
        "ordered_support_physical_ids_root_sha256_by_scenario": _scenario_map(
            f"{split_salt}:support-physical"
        ),
        "ordered_query_physical_ids_root_sha256_by_scenario": _scenario_map(
            f"{split_salt}:query-physical"
        ),
        "support_post_channel_iq_sha256_root_by_scenario": _scenario_map(
            f"{data_salt}:support-iq"
        ),
        "query_post_channel_iq_sha256_root_by_scenario": _scenario_map(
            f"{data_salt}:query-iq"
        ),
        "all_selected_physical_ids_root_sha256_by_scenario": _scenario_map(
            f"{data_salt}:all-selected-physical"
        ),
        "all_post_channel_iq_sha256_root_by_scenario": _scenario_map(
            f"{data_salt}:all-post-channel-iq"
        ),
        "same_scenario_support_query_physical_disjointness": "PASS",
        "cross_scenario_selected_physical_disjointness": "PASS",
        "cross_scenario_opaque_token_disjointness": "PASS",
        "single_leo_observation": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
        "query_decision_policy": "per_sample_all_registered_classes",
        "query_truth_in_predictor": False,
        "query_role_in_predictor": False,
    }
    return manifest, manifest_raw, seal, audit, audit_raw, commit


def _payload(**kwargs) -> dict:
    manifest, manifest_raw, seal, audit, audit_raw, commit = _control_payloads(
        **kwargs
    )
    seal_raw = canonical_json_bytes(seal) + b"\n"
    commit_raw = canonical_json_bytes(commit) + b"\n"
    return authority.build_phase2_data_authority_payload(
        predictor_manifest=manifest,
        predictor_manifest_raw=manifest_raw,
        predictor_seal=seal,
        predictor_seal_sha256=authority.sha256_bytes(seal_raw),
        offline_build_audit=audit,
        offline_build_audit_raw_sha256=authority.sha256_bytes(audit_raw),
        data_validation_commit=commit,
        data_validation_commit_sha256=authority.sha256_bytes(commit_raw),
    )


def _write_control_files(tmp_path: Path, **kwargs) -> dict[str, object]:
    manifest, manifest_raw, seal, audit, audit_raw, commit = _control_payloads(
        **kwargs
    )
    seal_raw = canonical_json_bytes(seal) + b"\n"
    commit_raw = canonical_json_bytes(commit) + b"\n"
    paths = {
        "manifest": tmp_path / "package_manifest.json",
        "seal": tmp_path / "predictor.seal.json",
        "audit": tmp_path / "offline_build_audit.json",
        "commit": tmp_path / "DATA_COMMIT.json",
    }
    paths["manifest"].write_bytes(manifest_raw)
    paths["seal"].write_bytes(seal_raw)
    paths["audit"].write_bytes(audit_raw)
    paths["commit"].write_bytes(commit_raw)
    return {
        **paths,
        "seal_sha": authority.sha256_bytes(seal_raw),
        "audit_sha": authority.sha256_bytes(audit_raw),
        "commit_sha": authority.sha256_bytes(commit_raw),
    }


def test_writer_opens_only_four_control_files_and_never_materializes_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _write_control_files(tmp_path)
    output = tmp_path / "data_authority.json"
    expected_reads = {
        Path(files[name]).resolve()
        for name in ("manifest", "seal", "audit", "commit")
    }
    opened_reads: list[Path] = []
    original_open = Path.open

    def guarded_open(path: Path, mode: str = "r", *args, **kwargs):
        resolved = path.resolve()
        if "r" in mode:
            if resolved not in expected_reads:
                raise AssertionError(f"unexpected control/payload read: {resolved}")
            opened_reads.append(resolved)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = authority.write_phase2_data_authority(
        predictor_manifest_path=files["manifest"],
        predictor_seal_path=files["seal"],
        expected_predictor_seal_sha256=files["seal_sha"],
        offline_build_audit_path=files["audit"],
        expected_offline_build_audit_sha256=files["audit_sha"],
        data_validation_commit_path=files["commit"],
        expected_data_validation_commit_sha256=files["commit_sha"],
        output_path=output,
    )
    monkeypatch.setattr(Path, "open", original_open)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == authority.DATA_AUTHORITY_STATUS
    assert payload["producer_access_audit"] == {
        "schema": authority.ACCESS_AUDIT_SCHEMA,
        "control_files_opened": [
            "predictor_manifest",
            "predictor_detached_seal",
            "offline_build_audit",
            "data_validation_commit_metadata",
        ],
        "support_payload_open_count": 0,
        "query_payload_open_count": 0,
        "iq_payload_materialized": False,
        "query_truth_open_count": 0,
        "cache_payload_open_count": 0,
        "data_revalidation_performed": False,
    }
    assert not any(tmp_path.glob("query_*.npz"))
    assert opened_reads == [
        Path(files[name]).resolve()
        for name in ("seal", "manifest", "audit", "commit")
    ]


def test_method_artifact_changes_do_not_change_capsule_or_split_id() -> None:
    first = _payload(method_salt="method-a")
    second = _payload(method_salt="method-b")
    assert first["source_binding"] != second["source_binding"]
    assert first["data_facts"]["final_registry"] != second["data_facts"][
        "final_registry"
    ]
    assert (
        first["data_facts"]["final_registry_identity_root_sha256"]
        == second["data_facts"]["final_registry_identity_root_sha256"]
    )
    assert first["capsule_id"] == second["capsule_id"]
    assert first["split_id"] == second["split_id"]
    identities = json.dumps(
        {"capsule": first["capsule_identity"], "split": first["split_identity"]},
        sort_keys=True,
    )
    for forbidden in ("candidate", "checkpoint", "adapter", "head", "package_root"):
        assert forbidden not in identities


def test_data_and_split_changes_change_their_method_free_identities() -> None:
    base = _payload(data_salt="data-a", split_salt="split-a")
    changed_data = _payload(data_salt="data-b", split_salt="split-a")
    changed_split = _payload(data_salt="data-a", split_salt="split-b")
    assert changed_data["capsule_id"] != base["capsule_id"]
    assert changed_split["capsule_id"] == base["capsule_id"]
    assert changed_split["split_id"] != base["split_id"]


def test_rng_seed_does_not_change_method_free_identities() -> None:
    first = _payload(seed=713101)
    second = _payload(seed=999999)
    assert first["capsule_id"] == second["capsule_id"]
    assert first["split_id"] == second["split_id"]


def test_stage2b_equal_registry_identity_is_valid() -> None:
    payload = _payload(stage="stage2b")
    assert payload["stage"] == "stage2b"
    assert payload["old_registry"] == payload["final_registry"]
    assert (
        payload["data_facts"]["old_registry_identity_root_sha256"]
        == payload["data_facts"]["final_registry_identity_root_sha256"]
    )


def test_registry_prefix_disjointness_and_single_observation_fail_closed() -> None:
    manifest, manifest_raw, seal, audit, audit_raw, commit = _control_payloads()
    seal_raw = canonical_json_bytes(seal) + b"\n"
    kwargs = {
        "predictor_manifest": manifest,
        "predictor_manifest_raw": manifest_raw,
        "predictor_seal": seal,
        "predictor_seal_sha256": authority.sha256_bytes(seal_raw),
        "offline_build_audit": audit,
        "offline_build_audit_raw_sha256": authority.sha256_bytes(audit_raw),
        "data_validation_commit_sha256": _sha("unused-integrity-pin"),
    }

    bad_prefix = dict(commit)
    bad_prefix["old_registry"] = list(reversed(commit["old_registry"]))
    with pytest.raises(authority.Phase2DataAuthorityError, match="registry prefix"):
        authority.build_phase2_data_authority_payload(
            **kwargs, data_validation_commit=bad_prefix
        )

    bad_disjoint = dict(commit)
    bad_disjoint["same_scenario_support_query_physical_disjointness"] = "FAIL"
    with pytest.raises(authority.Phase2DataAuthorityError, match="COMMIT drift"):
        authority.build_phase2_data_authority_payload(
            **kwargs, data_validation_commit=bad_disjoint
        )

    bad_audit = json.loads(json.dumps(audit))
    bad_audit["target_cache_audit"]["phase2_single_observation_compliant"] = False
    bad_audit_raw = canonical_json_bytes(bad_audit) + b"\n"
    with pytest.raises(authority.Phase2DataAuthorityError, match="single-observation"):
        authority.build_phase2_data_authority_payload(
            **{
                **kwargs,
                "offline_build_audit": bad_audit,
                "offline_build_audit_raw_sha256": authority.sha256_bytes(
                    bad_audit_raw
                ),
            },
            data_validation_commit=commit,
        )


def test_missing_upstream_root_fails_closed() -> None:
    manifest, manifest_raw, seal, audit, audit_raw, commit = _control_payloads()
    commit.pop("ordered_query_physical_ids_root_sha256_by_scenario")
    with pytest.raises(authority.Phase2DataAuthorityError, match="exact schema drift"):
        authority.build_phase2_data_authority_payload(
            predictor_manifest=manifest,
            predictor_manifest_raw=manifest_raw,
            predictor_seal=seal,
            predictor_seal_sha256=authority.sha256_bytes(
                canonical_json_bytes(seal) + b"\n"
            ),
            offline_build_audit=audit,
            offline_build_audit_raw_sha256=authority.sha256_bytes(audit_raw),
            data_validation_commit=commit,
            data_validation_commit_sha256=_sha("commit"),
        )


def test_expected_output_sha_is_integrity_only_and_cannot_make_formal(
    tmp_path: Path,
) -> None:
    files = _write_control_files(tmp_path)
    preview = _payload()
    expected_raw = canonical_json_bytes(preview) + b"\n"
    output = tmp_path / "pinned_data_authority.json"
    result = authority.write_phase2_data_authority(
        predictor_manifest_path=files["manifest"],
        predictor_seal_path=files["seal"],
        expected_predictor_seal_sha256=files["seal_sha"],
        offline_build_audit_path=files["audit"],
        expected_offline_build_audit_sha256=files["audit_sha"],
        data_validation_commit_path=files["commit"],
        expected_data_validation_commit_sha256=files["commit_sha"],
        output_path=output,
        expected_output_sha256=authority.sha256_bytes(expected_raw),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["output_sha256"] == authority.sha256_bytes(expected_raw)
    assert payload["schema"] == authority.DATA_AUTHORITY_SCHEMA
    assert payload["status"] == authority.DATA_AUTHORITY_STATUS
    assert payload["formal_data_authority"] is False
    assert payload["formal_launch_authority"] is False
    assert payload["formal_metric_claim_allowed"] is False
    assert payload["external_signature_present"] is False
    assert payload["external_signature_required_for_formal"] is True
    assert payload["phase2_data_status"] == "UPSTREAM_COMMIT_BLOCKED"
    assert payload["upstream_validated_once_claim_trusted"] is False
    assert payload["schema"] != "cvs.phase2.d81.external_row_authority.v1"


def test_commit_query_count_and_exact_k_are_enforced() -> None:
    manifest, manifest_raw, seal, audit, audit_raw, commit = _control_payloads()
    commit["support_count_by_scenario"] = {
        scenario: 14 for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    with pytest.raises(authority.Phase2DataAuthorityError, match="exact K"):
        authority.build_phase2_data_authority_payload(
            predictor_manifest=manifest,
            predictor_manifest_raw=manifest_raw,
            predictor_seal=seal,
            predictor_seal_sha256=authority.sha256_bytes(
                canonical_json_bytes(seal) + b"\n"
            ),
            offline_build_audit=audit,
            offline_build_audit_raw_sha256=authority.sha256_bytes(audit_raw),
            data_validation_commit=commit,
            data_validation_commit_sha256=_sha("commit"),
        )


def test_per_class_exact_k_rejects_balanced_total_but_unbalanced_classes() -> None:
    manifest, manifest_raw, seal, audit, audit_raw, commit = _control_payloads()
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    classes = list(commit["final_registry"])
    commit["support_count_by_class_by_scenario"][scenario][classes[0]] -= 1
    commit["support_count_by_class_by_scenario"][scenario][classes[1]] += 1
    with pytest.raises(authority.Phase2DataAuthorityError, match="class count"):
        authority.build_phase2_data_authority_payload(
            predictor_manifest=manifest,
            predictor_manifest_raw=manifest_raw,
            predictor_seal=seal,
            predictor_seal_sha256=authority.sha256_bytes(
                canonical_json_bytes(seal) + b"\n"
            ),
            offline_build_audit=audit,
            offline_build_audit_raw_sha256=authority.sha256_bytes(audit_raw),
            data_validation_commit=commit,
            data_validation_commit_sha256=_sha("commit"),
        )
