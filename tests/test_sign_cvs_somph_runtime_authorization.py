from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import somph_lineage_authority as authority
from cvsrffi import somph_predictor_bundle as predictor
from cvsrffi.leo_weak_cache import post_channel_iq_sha256
from scripts import sign_cvs_somph_runtime_authorization as bridge


def _openssl() -> Path:
    candidate = Path(bridge.lock_signer.PINNED_OPENSSL_BINARY_PATH)
    if not candidate.is_file():
        pytest.skip("pinned OpenSSL is required")
    return candidate


def _key(tmp_path: Path, name: str) -> Path:
    private = tmp_path / f"{name}.pem"
    subprocess.run(
        [str(_openssl()), "genpkey", "-algorithm", "ED25519", "-out", str(private)],
        check=True,
        capture_output=True,
    )
    return private


def _cache_and_packages() -> tuple[dict, dict[str, tuple]]:
    cache_by_scenario: dict = {}
    packages: dict[str, tuple] = {}
    all_tx = list(predictor.OLD_TX_IDS) + list(predictor.NEW_TX_IDS[:5])
    k_shot = 5
    scenario_payloads: dict[str, dict[str, np.ndarray]] = {}
    scenario_provenance: dict[str, dict[str, dict]] = {}
    for scenario_index, scenario in enumerate(predictor.FORMAL_LEO_WEAK_SCENARIOS):
        count = len(all_tx) * k_shot
        iq = (
            np.arange(count * 8, dtype=np.float32).reshape(count, 2, 4)
            + scenario_index * 100000
        )
        hashes = np.asarray([post_channel_iq_sha256(row) for row in iq])
        labels = np.repeat(np.arange(len(all_tx), dtype=np.int64), k_shot)
        ranks = np.tile(np.arange(k_shot, dtype=np.int64), len(all_tx))
        tx = np.repeat(np.asarray(all_tx), k_shot)
        roles = np.asarray(
            ["target_old" if index < 6 else "target_new" for index in labels]
        )
        seeds = np.arange(1000 + scenario_index * count, 1000 + (scenario_index + 1) * count, dtype=np.int64)
        physical = np.asarray(
            [f"physical-{scenario_index}-{index}" for index in range(count)]
        )
        overlays = np.asarray(
            [f"real-overlay-{scenario_index}-{index}" for index in range(count)]
        )
        cache_by_scenario[scenario] = {
            "leo_weak_iq": iq.copy(),
            "post_channel_iq_sha256": hashes.copy(),
            "satellite_seeds": seeds.copy(),
            "tx_ids": tx.copy(),
            "dataset_role": roles.copy(),
            "rx_ids": np.asarray(["20-1"] * count),
            "sat_scenarios": np.asarray([scenario] * count),
            "sample_ids": physical.copy(),
            "overlay_ids": overlays.copy(),
        }
        scenario_payloads[scenario] = {
            "support_leo_weak_iq": iq.copy(),
            "support_class_indices": labels.copy(),
            "support_rank_within_class": ranks.copy(),
            "support_tokens": np.asarray(
                [f"sid_{scenario_index:02x}{index:062x}" for index in range(count)]
            ),
            "support_overlay_tokens": np.asarray(
                [f"oid_{scenario_index:02x}{index:062x}" for index in range(count)]
            ),
            "support_satellite_seeds": seeds.copy(),
            "support_post_channel_iq_sha256": hashes.copy(),
        }
        scenario_provenance[scenario] = {
            str(token): {
                "sample_token": str(token),
                "scenario": scenario,
                "overlay_token": str(overlay),
                "satellite_seed": int(seed),
                "post_channel_iq_sha256": str(digest),
                "source_leo_cache_sha256": "a" * 64,
                "source_leo_provenance_sha256": "b" * 64,
            }
            for token, overlay, seed, digest in zip(
                scenario_payloads[scenario]["support_tokens"],
                scenario_payloads[scenario]["support_overlay_tokens"],
                seeds,
                hashes,
            )
        }

    registries = {
        state: [
            {"class_index": index, "class_handle": f"cls_{index:064x}"}
            for index in range(6 if state == "before" else 11)
        ]
        for state in ("before", "after")
    }
    for state in ("before", "after"):
        class_count = len(registries[state])
        row_count = class_count * k_shot
        payloads = {
            scenario: {
                key: np.array(value[:row_count], copy=True)
                for key, value in scenario_payloads[scenario].items()
            }
            for scenario in predictor.FORMAL_LEO_WEAK_SCENARIOS
        }
        provenance = {
            scenario: {
                token: scenario_provenance[scenario][token]
                for token in payloads[scenario]["support_tokens"].astype(str).tolist()
            }
            for scenario in predictor.FORMAL_LEO_WEAK_SCENARIOS
        }
        manifest = {
            "profile": predictor.ENROLLMENT_ONLY,
            "receiver": "20-1",
            "seed": 713101,
            "k_shot": k_shot,
            "support_pool_max_k": k_shot,
            "phase1_checkpoint_sha256": "c" * 64,
            "feature_runtime_sha256": "d" * 64,
            "method_lock_sha256": "e" * 64,
            "registration_state": state,
            "stage": "stage2b" if state == "before" else "stage2c",
            "registered_class_count": class_count,
            "registered_classes": registries[state],
            "package_root_sha256": ("1" if state == "before" else "2") * 64,
            "overlay_provenance_sha256": "3" * 64,
            **bridge.runtime_trust.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        }
        seal = {
            "artifact_member_allowlist_sha256": manifest["package_root_sha256"],
            "manifest_sha256": "4" * 64,
        }
        packages[state] = (manifest, seal, provenance, "5" * 64, payloads)
    return cache_by_scenario, packages


def _authority() -> tuple[dict, dict, dict]:
    scenarios = predictor.FORMAL_LEO_WEAK_SCENARIOS
    lock = {
        "receiver": "20-1",
        "seed": 713101,
        "cache_scope": "stage2_registered",
        "old_tx_ids": list(predictor.OLD_TX_IDS),
        "new_tx_ids": list(predictor.NEW_TX_IDS),
        "cache_set_manifest": {"path": "cache-set.json", "sha256": "6" * 64},
        "cache_sha256_by_scenario": {scenario: "a" * 64 for scenario in scenarios},
        "channel_config_sha256_by_scenario": {scenario: "7" * 64 for scenario in scenarios},
        "post_channel_iq_sha256_root_by_scenario": {scenario: "8" * 64 for scenario in scenarios},
        "overlay_ids_sha256_by_scenario": {scenario: "9" * 64 for scenario in scenarios},
    }
    attestation = {
        "structural_receipt_sha256": "b" * 64,
        "dataset_authority_root_sha256": "c" * 64,
        "cache_role_inputs_root_sha256": "d" * 64,
        "physical_sample_ids_sha256_by_scenario": {scenario: f"{index + 1:x}" * 64 for index, scenario in enumerate(scenarios)},
        "physical_sample_scenario_assignment_sha256": "e" * 64,
    }
    commit = {
        "authority_lock_sha256": "f" * 64,
        "members": [
            {"name": authority.CACHE_SPEC_MANIFEST_NAME, "sha256": "1" * 64},
            {"name": authority.AUTHORITY_ATTESTATION_NAME, "sha256": "2" * 64},
        ],
    }
    return lock, attestation, commit


def _install_prepare_stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    cache, packages = _cache_and_packages()
    lock, attestation, commit = _authority()

    def verify(_root, *, expected_commit_sha256):
        if expected_commit_sha256 != "3" * 64:
            raise authority.SomphLineageAuthorityError("external authority commit SHA mismatch")
        return lock, attestation, commit

    monkeypatch.setattr(bridge.authority, "verify_somph_lineage_authority_bundle", verify)
    monkeypatch.setattr(bridge, "_read_actual_formal_manifest", lambda *_a, **_k: ({"cell_count": 30}, "1" * 64))
    monkeypatch.setattr(bridge, "_read_policy", lambda _path: ({"schema": "v2"}, "4" * 64))
    monkeypatch.setattr(bridge, "_load_authority_bound_cache_set", lambda *_a, **_k: cache)

    def package(root, _seal):
        name = str(root)
        if "bad" in name:
            raise bridge.SomphRuntimeAuthorizationSigningError("package root mismatch")
        return packages["before" if "before" in name else "after"]

    monkeypatch.setattr(bridge, "_preflight_enrollment_package", package)
    return cache, packages


def _prepare(monkeypatch: pytest.MonkeyPatch):
    _install_prepare_stubs(monkeypatch)
    return bridge._prepare_formal_authorization_pair(
        actual_cache_manifest_path="manifest",
        authority_bundle_root="authority",
        expected_authority_commit_sha256="3" * 64,
        verified_cache_root="verified-cache",
        before_package_root="before-package",
        before_detached_seal_path="before-seal",
        after_package_root="after-package",
        after_detached_seal_path="after-seal",
        formal_policy_path="policy",
    )


def test_prepares_runtime_compatible_membership_bound_authorizations(monkeypatch):
    authorizations, evidence = _prepare(monkeypatch)
    assert evidence["selected_membership_roots"]["before"]["support_query_disjointness_status"] == "SUPPORT_ONLY_NO_QUERY_CLAIM"
    _, packages = _cache_and_packages()
    for state in ("before", "after"):
        auth = authorizations[state]
        assert auth["formal_launch_authority"] is True
        assert auth["formal_metric_claim_allowed"] is False
        predictor._validate_path_free_authorization_shape(auth)
        manifest, seal, provenance, seal_sha, _payloads = packages[state]
        roots = predictor._package_control_roots(
            manifest, provenance, new_tx_ids=list(predictor.NEW_TX_IDS[:5])
        )
        predictor._validate_path_free_formal_authorization(
            manifest=manifest,
            seal=seal,
            provenance_index=provenance,
            expected_package_detached_seal_sha256=seal_sha,
            authorization=auth,
            authorization_sha256=hashlib.sha256(predictor.canonical_json_bytes(auth)).hexdigest(),
            actual_formal_policy_sha256="4" * 64,
            code_closure_sha256=evidence["code_closure_sha256"],
            package_control_roots=roots,
        )


def test_production_entry_has_no_injectable_trust_or_runtime_root() -> None:
    parameters = inspect.signature(bridge.sign_runtime_authorization_pair).parameters
    assert not ({"public_key_hex", "public_key_sha256", "issuer", "key_id", "runtime_code_root"} & set(parameters))
    assert not hasattr(bridge, "_sign_runtime_authorization_pair_impl")
    members, root = bridge._runtime_code_closure()
    expected_members, expected_root = predictor._code_closure()
    assert members == expected_members
    assert root == expected_root


@pytest.mark.parametrize(
    "field",
    [
        "feature_runtime_sha256",
        "method_lock_sha256",
        "phase1_checkpoint_sha256",
        "registered_classes",
    ],
)
def test_before_after_runtime_and_old_registry_are_locked(field: str) -> None:
    _cache, packages = _cache_and_packages()
    before = copy.deepcopy(packages["before"][0])
    after = copy.deepcopy(packages["after"][0])
    if field == "registered_classes":
        after[field][0]["class_handle"] = "cls_" + "f" * 64
        message = "old registered-class prefix"
    else:
        after[field] = "f" * 64
        message = "row identity drift"
    with pytest.raises(bridge.SomphRuntimeAuthorizationSigningError, match=message):
        bridge._require_package_pair(before, after)


def test_verified_cache_roots_must_match_authority(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "cache-set.json"
    manifest_path.write_text("{}", encoding="utf-8")
    lock, attestation, _commit = _authority()
    scenarios = predictor.FORMAL_LEO_WEAK_SCENARIOS
    audit = {
        "physical_sample_ids_sha256_by_scenario": attestation[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": attestation[
            "physical_sample_scenario_assignment_sha256"
        ],
        "cache_audits": {
            scenario: {
                "sha256": lock["cache_sha256_by_scenario"][scenario],
                "physical_sample_ids_sha256": attestation[
                    "physical_sample_ids_sha256_by_scenario"
                ][scenario],
                "post_channel_iq_sha256_root": lock[
                    "post_channel_iq_sha256_root_by_scenario"
                ][scenario],
                "overlay_ids_sha256": lock["overlay_ids_sha256_by_scenario"][
                    scenario
                ],
            }
            for scenario in scenarios
        },
    }
    payload = {
        "cache_scope": "stage2_registered",
        "cache_sha256_by_scenario": lock["cache_sha256_by_scenario"],
    }
    monkeypatch.setattr(
        bridge.authority,
        "_read_external_json",
        lambda *_a, **_k: (payload, b"{}", "6" * 64, 2),
    )
    monkeypatch.setattr(
        bridge.authority,
        "_read_external_bytes",
        lambda *_a, **_k: (b"{}", "6" * 64, 2),
    )
    monkeypatch.setattr(
        bridge,
        "load_verified_leo_weak_cache_set",
        lambda *_a, **_k: ({scenario: {} for scenario in scenarios}, payload, audit),
    )
    bridge._load_authority_bound_cache_set(
        manifest_path, lock=lock, attestation=attestation
    )
    drift = copy.deepcopy(audit)
    drift["cache_audits"][scenarios[0]]["overlay_ids_sha256"] = "0" * 64
    monkeypatch.setattr(
        bridge,
        "load_verified_leo_weak_cache_set",
        lambda *_a, **_k: ({scenario: {} for scenario in scenarios}, payload, drift),
    )
    with pytest.raises(bridge.SomphRuntimeAuthorizationSigningError, match="roots drift"):
        bridge._load_authority_bound_cache_set(
            manifest_path, lock=lock, attestation=attestation
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_hash", "no unique verified-cache membership"),
        ("iq_bytes", "support IQ digest drift"),
        ("seed", "seed-TX-role-overlay binding drift"),
        ("tx", "seed-TX-role-overlay binding drift"),
        ("role", "seed-TX-role-overlay binding drift"),
        ("receiver", "seed-TX-role-overlay binding drift"),
        ("scenario", "seed-TX-role-overlay binding drift"),
        ("rank", "exact-K/rank assignment drift"),
        ("physical_overlap", "reuses physical samples across scenarios"),
    ],
)
def test_membership_negative_controls(mutation: str, message: str) -> None:
    cache, packages = _cache_and_packages()
    manifest, _seal, _provenance, _seal_sha, payloads = packages["before"]
    scenario = predictor.FORMAL_LEO_WEAK_SCENARIOS[0]
    if mutation == "duplicate_hash":
        cache[scenario]["post_channel_iq_sha256"][1] = cache[scenario]["post_channel_iq_sha256"][0]
    elif mutation == "iq_bytes":
        payloads[scenario]["support_leo_weak_iq"][0, 0, 0] += 1
    elif mutation == "seed":
        payloads[scenario]["support_satellite_seeds"][0] += 1
    elif mutation == "tx":
        cache[scenario]["tx_ids"][0] = predictor.NEW_TX_IDS[0]
    elif mutation == "role":
        cache[scenario]["dataset_role"][0] = "target_new"
    elif mutation == "receiver":
        cache[scenario]["rx_ids"][0] = "7-7"
    elif mutation == "scenario":
        cache[scenario]["sat_scenarios"][0] = predictor.FORMAL_LEO_WEAK_SCENARIOS[1]
    elif mutation == "rank":
        payloads[scenario]["support_rank_within_class"][0] = 4
    elif mutation == "physical_overlap":
        other = predictor.FORMAL_LEO_WEAK_SCENARIOS[1]
        cache[other]["sample_ids"][0] = cache[scenario]["sample_ids"][0]
    with pytest.raises(bridge.SomphRuntimeAuthorizationSigningError, match=message):
        bridge._verify_support_cache_membership(
            state="before",
            manifest=manifest,
            payloads=payloads,
            cache_arrays=cache,
            new_tx_ids=list(predictor.NEW_TX_IDS[:5]),
        )


def test_before_after_old_membership_must_be_stable() -> None:
    cache, packages = _cache_and_packages()
    before = bridge._verify_support_cache_membership(
        state="before", manifest=packages["before"][0], payloads=packages["before"][4], cache_arrays=cache, new_tx_ids=list(predictor.NEW_TX_IDS[:5])
    )[0]
    after_payloads = packages["after"][4]
    scenario = predictor.FORMAL_LEO_WEAK_SCENARIOS[0]
    for key in ("support_leo_weak_iq", "support_satellite_seeds", "support_post_channel_iq_sha256"):
        after_payloads[scenario][key][[0, 1]] = after_payloads[scenario][key][[1, 0]]
    after = bridge._verify_support_cache_membership(
        state="after", manifest=packages["after"][0], payloads=after_payloads, cache_arrays=cache, new_tx_ids=list(predictor.NEW_TX_IDS[:5])
    )[0]
    with pytest.raises(bridge.SomphRuntimeAuthorizationSigningError, match="old-class support membership drift"):
        bridge._require_before_after_old_membership_stable(before, after)


def test_wrong_manifest_and_package_fail_closed(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cells": []}), encoding="utf-8")
    with pytest.raises(bridge.SomphRuntimeAuthorizationSigningError, match="not the committed manifest"):
        bridge._read_actual_formal_manifest(manifest, expected_sha256="f" * 64, lock={"receiver": "20-1", "seed": 713101})
    _install_prepare_stubs(monkeypatch)
    with pytest.raises(bridge.SomphRuntimeAuthorizationSigningError, match="package root mismatch"):
        bridge._prepare_formal_authorization_pair(
            actual_cache_manifest_path="manifest", authority_bundle_root="authority", expected_authority_commit_sha256="3" * 64, verified_cache_root="cache", before_package_root="bad-before", before_detached_seal_path="seal", after_package_root="after", after_detached_seal_path="seal", formal_policy_path="policy"
        )


def test_wrong_private_key_cannot_publish_formal_outputs(tmp_path: Path, monkeypatch) -> None:
    authorizations, evidence = _prepare(monkeypatch)
    monkeypatch.setattr(bridge, "_prepare_formal_authorization_pair", lambda **_kwargs: (authorizations, evidence))
    private = _key(tmp_path, "wrong")
    output = tmp_path / "formal-output"
    with pytest.raises(bridge.SomphRuntimeAuthorizationSigningError, match="signature invalid"):
        bridge.sign_runtime_authorization_pair(
            actual_cache_manifest_path="manifest", authority_bundle_root="authority", expected_authority_commit_sha256="3" * 64, verified_cache_root="cache", before_package_root="before", before_detached_seal_path="seal", after_package_root="after", after_detached_seal_path="seal", formal_policy_path="policy", private_key_path=private, openssl_bin=_openssl(), output_root=output
        )
    assert not output.exists()


def test_publish_rename_failure_cleans_staging(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "published"
    monkeypatch.setattr(bridge.os, "rename", lambda *_args: (_ for _ in ()).throw(OSError("rename failed")))
    with pytest.raises(OSError, match="rename failed"):
        bridge._publish_output_root(output, {"one.json": b"{}\n"})
    assert not output.exists()
    assert not list(tmp_path.glob(".published.staging-*"))


def test_publish_parent_fsync_failure_rolls_back(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "published"
    real_fsync = bridge.lock_signer._fsync_directory

    def fail_after_publish(path: Path) -> None:
        if Path(path) == tmp_path and output.exists():
            raise OSError("parent fsync failed")
        real_fsync(path)

    monkeypatch.setattr(bridge.lock_signer, "_fsync_directory", fail_after_publish)
    with pytest.raises(OSError, match="parent fsync failed"):
        bridge._publish_output_root(output, {"one.json": b"{}\n"})
    assert not output.exists()
    assert not list(tmp_path.glob(".published.staging-*"))
