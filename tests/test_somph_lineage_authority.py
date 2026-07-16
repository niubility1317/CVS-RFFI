from __future__ import annotations

import json
import hashlib
import secrets
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.somph_lineage_authority as authority
from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA,
    LEO_WEAK_CACHE_SET_SCHEMA,
    LEO_WEAK_CACHE_STAGE,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
    physical_sample_id_from_values,
    post_channel_iq_sha256,
    sha256_file,
)
from cvsrffi.somph_formal_matrix import DEVELOPMENT_SEED, OLD_TX_IDS
from cvsrffi.somph_leo_weak_lineage_seal import CHANNEL_CODE_CLOSURE_SCHEMA
from cvsrffi.somph_lineage_authority import (
    AUTHORITY_LOCK_SCHEMA,
    SomphLineageAuthorityError,
    verify_somph_lineage_authority_bundle,
    write_somph_lineage_authority_bundle,
)
from cvsrffi.stage2_predictor_bundle import canonical_json_bytes, sha256_bytes


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _descriptor(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _signed_envelope(
    lock: dict,
    *,
    build_receipt_sha256: str = "b" * 64,
) -> tuple[dict, bytes]:
    seed = secrets.token_bytes(32)
    hashed = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(
        bytes([hashed[0] & 248])
        + hashed[1:31]
        + bytes([(hashed[31] & 63) | 64]),
        "little",
    )
    public_key = authority._ed_encode(
        authority._ed_scalar_mult(authority._ED_B, scalar)
    )
    envelope = {
        "schema": "cvs.phase2.somph_leo_weak_signed_authority_envelope.v1",
        "domain": "cvs.somph.leo_weak.authority_lock.ed25519.v1",
        "issuer": "qknnv42_stage2bc_extreme_light_route_20260716",
        "key_id": "somph-authority-ed25519-20260716",
        "lock_canonical_sha256": sha256_bytes(canonical_json_bytes(lock)),
        "authority_lock_build_receipt_sha256": build_receipt_sha256,
        "cache_spec_manifest_sha256": (
            "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
        ),
        "cache_spec_cell_id": (
            f"rx_{str(lock['receiver']).replace('-', '_')}_seed_{lock['seed']}"
        ),
        "signature_ed25519_hex": "",
    }
    message = authority._authority_signature_message(envelope)
    nonce = int.from_bytes(
        hashlib.sha512(hashed[32:] + message).digest(), "little"
    ) % authority._ED_L
    encoded_r = authority._ed_encode(
        authority._ed_scalar_mult(authority._ED_B, nonce)
    )
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public_key + message).digest(), "little"
    ) % authority._ED_L
    signature_scalar = (nonce + challenge * scalar) % authority._ED_L
    envelope["signature_ed25519_hex"] = (
        encoded_r + signature_scalar.to_bytes(32, "little")
    ).hex()
    return envelope, public_key


def _install_test_envelope_verifier(
    monkeypatch: pytest.MonkeyPatch,
    public_key: bytes,
) -> None:
    def verify(
        envelope: dict,
        *,
        lock_canonical_sha256: str,
        expected_cache_spec_cell_id: str,
        expected_build_receipt_sha256: str | None = None,
    ) -> None:
        expected = {
            "schema": (
                "cvs.phase2.somph_leo_weak_signed_authority_envelope.v1"
            ),
            "domain": "cvs.somph.leo_weak.authority_lock.ed25519.v1",
            "issuer": "qknnv42_stage2bc_extreme_light_route_20260716",
            "key_id": "somph-authority-ed25519-20260716",
            "lock_canonical_sha256": lock_canonical_sha256,
            "authority_lock_build_receipt_sha256": (
                expected_build_receipt_sha256 or "b" * 64
            ),
            "cache_spec_manifest_sha256": (
                "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
            ),
            "cache_spec_cell_id": expected_cache_spec_cell_id,
        }
        if any(envelope.get(key) != value for key, value in expected.items()):
            raise SomphLineageAuthorityError(
                "test authority envelope identity/binding drift"
            )
        authority._verify_ed25519(
            public_key,
            authority._authority_signature_message(envelope),
            bytes.fromhex(envelope["signature_ed25519_hex"]),
        )

    monkeypatch.setattr(authority, "_verify_signed_envelope", verify)


def _install_test_build_authority_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verify(
        _lock: dict,
        *,
        build_receipt_sha256: str,
        **_kwargs,
    ) -> tuple[str, str]:
        receipt_sha = authority._require_sha256(
            build_receipt_sha256,
            field="test build receipt SHA",
        )
        receiver, seed, roles = authority._validate_lock_formal_identity(_lock)
        return receipt_sha, authority._dataset_root_from_lock(
            _lock,
            receiver=receiver,
            seed=seed,
            roles=roles,
        )

    monkeypatch.setattr(authority, "_verify_build_authority_binding", verify)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    receiver = "20-1"
    seed = DEVELOPMENT_SEED
    dataset = tmp_path / "ManySig.pkl"
    dataset.write_bytes(b"authority-dataset-bytes")
    exporter = tmp_path / "build_cvs_leo_weak_iq_cache.py"
    exporter.write_text("print('offline exporter')\n", encoding="utf-8")

    channel_members = []
    channel_paths = {}
    for index, logical_name in enumerate(authority.CHANNEL_CODE_LOGICAL_MEMBERS):
        path = tmp_path / f"channel_{index}.py"
        path.write_text(f"# {logical_name}\nVALUE = {index}\n", encoding="utf-8")
        channel_paths[logical_name] = path
        channel_members.append(
            {
                "logical_name": logical_name,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    channel_closure_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "schema": CHANNEL_CODE_CLOSURE_SCHEMA,
                "members": [
                    {
                        "logical_name": item["logical_name"],
                        "sha256": item["sha256"],
                        "size_bytes": item["size_bytes"],
                    }
                    for item in channel_members
                ],
            }
        )
    )

    old_tx = list(OLD_TX_IDS)
    role_spec = {
        "role": "target_old",
        "pkl": str(dataset),
        "tx_ids": ",".join(old_tx),
        "rxs": receiver,
        "days": "0",
        "max_samples_per_tx": 40,
    }
    cache_paths = {
        scenario: str(tmp_path / f"{scenario}.npz")
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    build_spec = {
        "schema": "cvs_leo_weak_iq_cache_build_spec_v2",
        "cache_set_id": "authority-test",
        "cache_scope": "stage2_target_old",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        "physical_sample_scenario_assignment_policy": (
            authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "star_ground_channel_impl": "simplified_leo_residual",
        "role_specs": [role_spec],
        "dataset_seed": seed,
        "satellite_seed_by_scenario": {
            scenario: seed * 10 + index
            for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
        },
        "out_npz_by_scenario": cache_paths,
        "out_manifest": str(tmp_path / "cache_set.json"),
        "batch_size": 256,
        "wisig_out_len": 256,
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
    }
    build_spec_path = tmp_path / "build_spec.json"
    _write_json(build_spec_path, build_spec)
    build_spec_canonical_sha = canonical_json_sha256(build_spec)
    exporter_sha = sha256_file(exporter)

    dataset_sha = sha256_file(dataset)
    cache_hashes = {}
    physical_roots = {}
    physical_ids_by_scenario = {}
    iq_roots = {}
    overlay_roots = {}
    channel_config_hashes = {}
    cache_audits = {}
    row_count = len(old_tx)
    authority_role_inputs = None
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        record_indices = np.asarray(
            [scenario_index * 1000 + index for index in range(row_count)],
            dtype=np.int64,
        )
        sample_ids = [
            physical_sample_id_from_values(
                dataset_sha256=dataset_sha,
                source_record_index=int(record_indices[index]),
                role="target_old",
                tx_id=tx_id,
                rx_id=receiver,
                day_id="0",
                eq_id="1",
                sig_id=str(scenario_index * 1000 + index),
            )
            for index, tx_id in enumerate(old_tx)
        ]
        physical_root = ids_sha256(sample_ids)
        iq = (
            np.arange(row_count * 2 * 4, dtype=np.float32).reshape(
                row_count, 2, 4
            )
            + scenario_index
        )
        satellite_seed = seed * 10 + scenario_index
        seeds = np.full(row_count, satellite_seed, dtype=np.int64)
        iq_hashes = [post_channel_iq_sha256(row) for row in iq]
        channel_config = {
            "channel_model": "leo_residual",
            "scenario": scenario,
            "star_ground_channel_impl": "simplified_leo_residual",
        }
        channel_hash = canonical_json_sha256(channel_config)
        overlays = [
            overlay_id(
                sample_id=sample_id,
                scenario=scenario,
                satellite_seed=satellite_seed,
                channel_config_sha256=channel_hash,
                iq_sha256=iq_hash,
            )
            for sample_id, iq_hash in zip(sample_ids, iq_hashes)
        ]
        role_input = {
            "role": "target_old",
            "dataset_sha256": sha256_file(dataset),
            "dataset_size_bytes": dataset.stat().st_size,
            "requested_tx_ids": ",".join(old_tx),
            "requested_rxs": receiver,
            "requested_days": "0",
            "dataset_seed": seed,
            "resolved_info": {"fixture": True},
            "physical_sample_count": row_count,
        }
        authority_role_inputs = [role_input]
        manifest = {
            "schema": LEO_WEAK_CACHE_SCHEMA,
            "artifact_stage": LEO_WEAK_CACHE_STAGE,
            "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "contains_post_channel_iq_only": True,
            "contains_clean_rows": False,
            "target_channel_view": "leo_weak_only",
            "target_channel_scenarios": [scenario],
            "scenario": scenario,
            "iq_array_key": "leo_weak_iq",
            "raw_or_clean_iq_key_present": False,
            "overlay_applied_before_phase2": True,
            "star_ground_channel_impl": "simplified_leo_residual",
            "channel_model": "leo_residual",
            "channel_config": channel_config,
            "channel_config_sha256": channel_hash,
            "builder_sha256": exporter_sha,
            "build_spec_sha256": build_spec_canonical_sha,
            "output_roles": ["target_old"],
            "role_satellite_seeds": {"target_old": satellite_seed},
            "role_inputs": [role_input],
            "row_count": row_count,
            "physical_sample_ids_sha256": physical_root,
            "post_channel_iq_sha256_root": ids_sha256(iq_hashes),
            "overlay_ids_sha256": ids_sha256(overlays),
            "channel_meta_keys": ["channel_model"],
            "sample_overlay_provenance_fields": [
                "sample_ids",
                "source_dataset_sha256",
                "source_record_indices",
                "sat_scenarios",
                "satellite_seeds",
                "post_channel_iq_sha256",
                "overlay_ids",
            ],
            **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
            "physical_sample_scenario_assignment_policy": (
                authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
            ),
        }
        cache = Path(cache_paths[scenario])
        with cache.open("xb") as handle:
            np.savez(
                handle,
                leo_weak_iq=iq,
                raw_labels=np.arange(row_count, dtype=np.int64),
                domain_labels=np.zeros(row_count, dtype=np.int64),
                tx_ids=np.asarray(old_tx),
                rx_ids=np.asarray([receiver] * row_count),
                day_ids=np.asarray(["0"] * row_count),
                eq_ids=np.asarray(["1"] * row_count),
                sig_ids=np.asarray(
                    [
                        str(scenario_index * 1000 + index)
                        for index in range(row_count)
                    ]
                ),
                source_dataset_sha256=np.asarray([dataset_sha] * row_count),
                source_record_indices=record_indices,
                dataset_role=np.asarray(["target_old"] * row_count),
                channel_views=np.asarray(["rx_base"] * row_count),
                sat_scenarios=np.asarray([scenario] * row_count),
                satellite_seeds=seeds,
                overlay_applied=np.asarray([True] * row_count, dtype=bool),
                sample_ids=np.asarray(sample_ids),
                post_channel_iq_sha256=np.asarray(iq_hashes),
                overlay_ids=np.asarray(overlays),
                manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
            )
        cache_hashes[scenario] = sha256_file(cache)
        physical_roots[scenario] = physical_root
        physical_ids_by_scenario[scenario] = sample_ids
        iq_roots[scenario] = ids_sha256(iq_hashes)
        overlay_roots[scenario] = ids_sha256(overlays)
        channel_config_hashes[scenario] = channel_hash
        cache_audits[scenario] = {}

    assignment_root = canonical_json_sha256(physical_ids_by_scenario)
    cache_set = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_set_id": "authority-test",
        "cache_scope": "stage2_target_old",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": ["target_old"],
        "cache_npz_by_scenario": {
            scenario: Path(cache_paths[scenario]).name
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_sha256_by_scenario": cache_hashes,
        "cache_audits": cache_audits,
        **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        "physical_sample_scenario_assignment_policy": (
            authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": physical_roots,
        "physical_sample_scenario_assignment_sha256": assignment_root,
        "builder_sha256": exporter_sha,
        "build_spec_sha256": build_spec_canonical_sha,
        "build_spec_path_exposed_to_phase2": False,
    }
    cache_set_path = tmp_path / "cache_set.json"
    _write_json(cache_set_path, cache_set)

    lock = {
        "schema": AUTHORITY_LOCK_SCHEMA,
        "receiver": receiver,
        "seed": seed,
        "cache_scope": "stage2_target_old",
        "old_tx_ids": old_tx,
        "new_tx_ids": [],
        "cache_set_manifest": _descriptor(cache_set_path),
        "cache_sha256_by_scenario": cache_hashes,
        "exporter": _descriptor(exporter),
        "build_spec": {
            "path": str(build_spec_path),
            "file_sha256": sha256_file(build_spec_path),
            "canonical_sha256": build_spec_canonical_sha,
            "size_bytes": build_spec_path.stat().st_size,
        },
        "channel_code_closure": {
            "closure_sha256": channel_closure_sha,
            "members": channel_members,
        },
        "channel_config_sha256_by_scenario": channel_config_hashes,
        **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        "physical_sample_scenario_assignment_policy": (
            authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": physical_roots,
        "physical_sample_scenario_assignment_sha256": assignment_root,
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "post_channel_iq_sha256_root_by_scenario": iq_roots,
        "overlay_ids_sha256_by_scenario": overlay_roots,
        "cache_role_inputs_root_sha256": sha256_bytes(
            canonical_json_bytes(authority_role_inputs)
        ),
        "datasets": [
            {
                "role": "target_old",
                "path": str(dataset),
                "sha256": sha256_file(dataset),
                "size_bytes": dataset.stat().st_size,
                "tx_ids": old_tx,
            }
        ],
    }
    lock_path = tmp_path / "authority_lock.json"
    _write_json(lock_path, lock)
    build_receipt_path = tmp_path / "authority_lock_build_receipt.json"
    _write_json(build_receipt_path, {"fixture": True})
    cache_spec_manifest_path = tmp_path / "cache_spec_manifest.json"
    _write_json(cache_spec_manifest_path, {"fixture": True})
    build_receipt_sha = sha256_file(build_receipt_path)
    envelope, public_key = _signed_envelope(
        lock,
        build_receipt_sha256=build_receipt_sha,
    )
    _install_test_envelope_verifier(monkeypatch, public_key)
    _install_test_build_authority_verifier(monkeypatch)
    envelope_path = tmp_path / "signed_authority_envelope.json"
    _write_json(envelope_path, envelope)
    return {
        "authority_lock_path": lock_path,
        "signed_authority_envelope_path": envelope_path,
        "expected_signed_authority_envelope_sha256": sha256_file(envelope_path),
        "authority_lock_build_receipt_path": build_receipt_path,
        "cache_spec_manifest_path": cache_spec_manifest_path,
        "output_root": tmp_path / "authority_bundle",
        "dataset": dataset,
        "build_spec_path": build_spec_path,
        "lock": lock,
    }


def _rewrite_lock(kwargs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_json(kwargs["authority_lock_path"], kwargs["lock"])
    envelope, public_key = _signed_envelope(
        kwargs["lock"],
        build_receipt_sha256=sha256_file(
            kwargs["authority_lock_build_receipt_path"]
        ),
    )
    _install_test_envelope_verifier(monkeypatch, public_key)
    _write_json(kwargs["signed_authority_envelope_path"], envelope)
    kwargs["expected_signed_authority_envelope_sha256"] = sha256_file(
        kwargs["signed_authority_envelope_path"]
    )


def _write_bundle(kwargs: dict, *, expected_envelope_sha: str | None = None):
    return write_somph_lineage_authority_bundle(
        kwargs["authority_lock_path"],
        signed_authority_envelope_path=kwargs[
            "signed_authority_envelope_path"
        ],
        expected_signed_authority_envelope_sha256=(
            expected_envelope_sha
            if expected_envelope_sha is not None
            else kwargs["expected_signed_authority_envelope_sha256"]
        ),
        authority_lock_build_receipt_path=kwargs[
            "authority_lock_build_receipt_path"
        ],
        cache_spec_manifest_path=kwargs["cache_spec_manifest_path"],
        output_root=kwargs["output_root"],
    )


def test_pure_python_ed25519_verifier_matches_rfc8032_vector() -> None:
    public_key = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a"
        "0ee172f3daa62325af021a68f707511a"
    )
    signature = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a"
        "84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46b"
        "d25bf5f0595bbe24655141438e7a100b"
    )
    authority._verify_ed25519(public_key, b"", signature)
    with pytest.raises(
        SomphLineageAuthorityError,
        match="Ed25519 authority signature",
    ):
        authority._verify_ed25519(public_key, b"x", signature)


def test_authority_bundle_cli_requires_complete_build_evidence() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "build_cvs_somph_lineage_authority_bundle.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--authority-lock",
        "--signed-authority-envelope",
        "--expected-signed-authority-envelope-sha256",
        "--authority-lock-build-receipt",
        "--cache-spec-manifest",
        "--output-root",
    ):
        assert option in completed.stdout


def test_production_identity_ignores_mutated_display_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authority, "PINNED_AUTHORITY_ISSUER", "attacker")
    monkeypatch.setattr(authority, "PINNED_AUTHORITY_KEY_ID", "attacker")
    monkeypatch.setattr(
        authority, "PINNED_AUTHORITY_PUBLIC_KEY_HEX", "00" * 32
    )
    monkeypatch.setattr(
        authority, "PINNED_AUTHORITY_PUBLIC_KEY_SHA256", "00" * 32
    )
    envelope = {
        "schema": (
            "cvs.phase2.somph_leo_weak_signed_authority_envelope.v1"
        ),
        "domain": "cvs.somph.leo_weak.authority_lock.ed25519.v1",
        "issuer": "qknnv42_stage2bc_extreme_light_route_20260716",
        "key_id": "somph-authority-ed25519-20260716",
        "lock_canonical_sha256": "1" * 64,
        "authority_lock_build_receipt_sha256": "b" * 64,
        "cache_spec_manifest_sha256": (
            "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
        ),
        "cache_spec_cell_id": "rx_20_1_seed_713101",
        "signature_ed25519_hex": "00" * 64,
    }
    with pytest.raises(
        SomphLineageAuthorityError,
        match="Ed25519",
    ):
        authority._verify_signed_envelope(
            envelope,
            lock_canonical_sha256="1" * 64,
            expected_cache_spec_cell_id="rx_20_1_seed_713101",
            expected_build_receipt_sha256="b" * 64,
        )


def test_attestation_and_commit_ignore_mutated_display_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, "PINNED_AUTHORITY_ISSUER", "attacker")
    monkeypatch.setattr(authority, "PINNED_AUTHORITY_KEY_ID", "attacker")
    result = _write_bundle(kwargs)
    attestation = json.loads(
        (
            kwargs["output_root"] / authority.AUTHORITY_ATTESTATION_NAME
        ).read_text(encoding="utf-8")
    )
    commit = json.loads(
        (kwargs["output_root"] / authority.AUTHORITY_COMMIT_NAME).read_text(
            encoding="utf-8"
        )
    )
    for payload in (attestation, commit):
        assert payload["authority_issuer"] == (
            "qknnv42_stage2bc_extreme_light_route_20260716"
        )
        assert payload["authority_key_id"] == (
            "somph-authority-ed25519-20260716"
        )
    assert result["external_authority_lock_verified"] is True


def test_publishes_and_consumes_from_external_commit_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    result = _write_bundle(kwargs)
    assert result["external_authority_lock_verified"] is True
    assert result["formal_launch_authority"] is False
    lock, attestation, commit = verify_somph_lineage_authority_bundle(
        kwargs["output_root"],
        expected_commit_sha256=result["authority_commit_sha256"],
    )
    assert lock["receiver"] == "20-1"
    assert attestation["external_authority_lock_verified"] is True
    assert attestation["formal_launch_authority"] is False
    assert commit["formal_launch_authority"] is False
    assert not list(tmp_path.glob(".authority_bundle.*.staging"))


def test_rejects_nonexternal_envelope_sha_and_dataset_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    with pytest.raises(
        SomphLineageAuthorityError,
        match="external signed authority envelope SHA",
    ):
        _write_bundle(kwargs, expected_envelope_sha="f" * 64)
    kwargs["dataset"].write_bytes(b"tampered dataset")
    with pytest.raises(
        SomphLineageAuthorityError, match="external byte binding"
    ):
        _write_bundle(kwargs)


def test_rejects_self_signed_envelope_from_nonpinned_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    rogue_envelope, _rogue_public_key = _signed_envelope(
        kwargs["lock"],
        build_receipt_sha256=sha256_file(
            kwargs["authority_lock_build_receipt_path"]
        ),
    )
    _write_json(kwargs["signed_authority_envelope_path"], rogue_envelope)
    kwargs["expected_signed_authority_envelope_sha256"] = sha256_file(
        kwargs["signed_authority_envelope_path"]
    )
    with pytest.raises(
        SomphLineageAuthorityError,
        match="Ed25519 authority signature",
    ):
        _write_bundle(kwargs)


def test_rejects_authority_lock_extra_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    kwargs["lock"]["manifest_sha256"] = "a" * 64
    _rewrite_lock(kwargs, monkeypatch)
    with pytest.raises(SomphLineageAuthorityError, match="lock exact schema"):
        _write_bundle(kwargs)


def test_rejects_build_spec_semantic_drift_and_channel_allowlist_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path / "build", monkeypatch)
    build_spec = json.loads(kwargs["build_spec_path"].read_text(encoding="utf-8"))
    build_spec["query_labels"] = ["forbidden"]
    _write_json(kwargs["build_spec_path"], build_spec)
    kwargs["lock"]["build_spec"].update(
        {
            "file_sha256": sha256_file(kwargs["build_spec_path"]),
            "canonical_sha256": canonical_json_sha256(build_spec),
            "size_bytes": kwargs["build_spec_path"].stat().st_size,
        }
    )
    _rewrite_lock(kwargs, monkeypatch)
    with pytest.raises(SomphLineageAuthorityError, match="build spec exact schema"):
        _write_bundle(kwargs)

    kwargs = _fixture(tmp_path / "channel", monkeypatch)
    kwargs["lock"]["channel_code_closure"]["members"][0][
        "logical_name"
    ] = "../clean/channel.py"
    _rewrite_lock(kwargs, monkeypatch)
    with pytest.raises(
        SomphLineageAuthorityError, match="logical member allowlist"
    ):
        _write_bundle(kwargs)


def test_rejects_cache_tx_coverage_not_matching_locked_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    cache = tmp_path / f"{scenario}.npz"
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["tx_ids"][0] = payload["tx_ids"][1]
    cache.unlink()
    with cache.open("xb") as handle:
        np.savez(handle, **payload)
    new_cache_sha = sha256_file(cache)
    cache_set_path = Path(kwargs["lock"]["cache_set_manifest"]["path"])
    cache_set = json.loads(cache_set_path.read_text(encoding="utf-8"))
    cache_set["cache_sha256_by_scenario"][scenario] = new_cache_sha
    _write_json(cache_set_path, cache_set)
    kwargs["lock"]["cache_sha256_by_scenario"][scenario] = new_cache_sha
    kwargs["lock"]["cache_set_manifest"] = _descriptor(cache_set_path)
    _rewrite_lock(kwargs, monkeypatch)
    with pytest.raises(SomphLineageAuthorityError, match="TX coverage"):
        _write_bundle(kwargs)


def test_rejects_cache_role_input_dataset_self_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    cache = tmp_path / f"{scenario}.npz"
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    manifest = json.loads(str(payload["manifest_json"].reshape(-1)[0]))
    manifest["role_inputs"][0]["dataset_sha256"] = "f" * 64
    payload["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    cache.unlink()
    with cache.open("xb") as handle:
        np.savez(handle, **payload)
    new_cache_sha = sha256_file(cache)
    cache_set_path = Path(kwargs["lock"]["cache_set_manifest"]["path"])
    cache_set = json.loads(cache_set_path.read_text(encoding="utf-8"))
    cache_set["cache_sha256_by_scenario"][scenario] = new_cache_sha
    _write_json(cache_set_path, cache_set)
    kwargs["lock"]["cache_sha256_by_scenario"][scenario] = new_cache_sha
    kwargs["lock"]["cache_set_manifest"] = _descriptor(cache_set_path)
    _rewrite_lock(kwargs, monkeypatch)
    with pytest.raises(
        SomphLineageAuthorityError, match="role_inputs authority binding"
    ):
        _write_bundle(kwargs)


def test_rejects_signed_cache_role_inputs_root_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    kwargs["lock"]["cache_role_inputs_root_sha256"] = "f" * 64
    _rewrite_lock(kwargs, monkeypatch)
    with pytest.raises(
        SomphLineageAuthorityError,
        match="role-input root mismatch",
    ):
        _write_bundle(kwargs)


def test_consumer_rejects_tampered_committed_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    result = _write_bundle(kwargs)
    attestation = kwargs["output_root"] / authority.AUTHORITY_ATTESTATION_NAME
    os.chmod(attestation, 0o600)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["formal_launch_authority"] = True
    _write_json(attestation, payload)
    with pytest.raises(
        SomphLineageAuthorityError, match="committed member mismatch"
    ):
        verify_somph_lineage_authority_bundle(
            kwargs["output_root"],
            expected_commit_sha256=result["authority_commit_sha256"],
        )


def test_consumer_rejects_recommitted_build_receipt_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    _write_bundle(kwargs)
    root = kwargs["output_root"]
    os.chmod(root, 0o755)
    receipt_path = root / authority.AUTHORITY_LOCK_BUILD_RECEIPT_NAME
    commit_path = root / authority.AUTHORITY_COMMIT_NAME
    os.chmod(receipt_path, 0o600)
    os.chmod(commit_path, 0o600)
    _write_json(receipt_path, {"fixture": "substituted"})
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    for descriptor in commit["members"]:
        if descriptor["name"] == authority.AUTHORITY_LOCK_BUILD_RECEIPT_NAME:
            descriptor["sha256"] = sha256_file(receipt_path)
            descriptor["size_bytes"] = receipt_path.stat().st_size
    commit["bundle_root_sha256"] = sha256_bytes(
        canonical_json_bytes(commit["members"])
    )
    _write_json(commit_path, commit)
    substituted_commit_sha = sha256_file(commit_path)
    os.chmod(receipt_path, 0o444)
    os.chmod(commit_path, 0o444)
    os.chmod(root, 0o555)

    with pytest.raises(
        SomphLineageAuthorityError,
        match="identity/binding drift",
    ):
        verify_somph_lineage_authority_bundle(
            root,
            expected_commit_sha256=substituted_commit_sha,
        )


def test_consumer_starts_from_external_expected_commit_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    _write_bundle(kwargs)
    with pytest.raises(
        SomphLineageAuthorityError, match="external authority commit SHA"
    ):
        verify_somph_lineage_authority_bundle(
            kwargs["output_root"],
            expected_commit_sha256="f" * 64,
        )


def test_staging_is_rolled_back_if_attestation_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    original = authority._write_new_readonly

    def fail_attestation(path: Path, payload: bytes):
        if path.name == authority.AUTHORITY_ATTESTATION_NAME:
            raise OSError("injected attestation failure")
        return original(path, payload)

    monkeypatch.setattr(authority, "_write_new_readonly", fail_attestation)
    with pytest.raises(OSError, match="injected attestation failure"):
        _write_bundle(kwargs)
    assert not kwargs["output_root"].exists()
    assert not list(tmp_path.glob(".authority_bundle.*.staging"))


def test_bundle_is_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    _write_bundle(kwargs)
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_bundle(kwargs)
