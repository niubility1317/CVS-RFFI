from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cvsrffi import somph_lineage_authority as authority
from scripts import sign_cvs_somph_authority_lock as signer


def _openssl() -> Path:
    executable = Path(signer.PINNED_OPENSSL_BINARY_PATH)
    if not executable.is_file():
        pytest.skip("OpenSSL is required for authority signing tests")
    return executable.resolve()


def _generate_key(tmp_path: Path, name: str) -> tuple[Path, bytes]:
    private_key = tmp_path / f"{name}.pem"
    subprocess.run(
        [
            str(_openssl()),
            "genpkey",
            "-algorithm",
            "ED25519",
            "-out",
            str(private_key),
        ],
        check=True,
        capture_output=True,
    )
    public_der = subprocess.run(
        [
            str(_openssl()),
            "pkey",
            "-in",
            str(private_key),
            "-pubout",
            "-outform",
            "DER",
        ],
        check=True,
        capture_output=True,
    ).stdout
    prefix = bytes.fromhex("302a300506032b6570032100")
    assert public_der.startswith(prefix)
    assert len(public_der) == len(prefix) + 32
    return private_key, public_der[-32:]


def _test_verifier(public_key: bytes):
    expected_identity = {
        "schema": authority.AUTHORITY_ENVELOPE_SCHEMA,
        "domain": authority.AUTHORITY_SIGNATURE_DOMAIN,
        "issuer": authority.PINNED_AUTHORITY_ISSUER,
        "key_id": authority.PINNED_AUTHORITY_KEY_ID,
    }

    def verify_test_envelope(
        envelope: dict,
        *,
        lock_canonical_sha256: str,
        expected_cache_spec_cell_id: str,
        expected_build_receipt_sha256: str | None = None,
    ) -> None:
        if set(envelope) != authority._ENVELOPE_KEYS:
            raise authority.SomphLineageAuthorityError(
                "signed authority envelope exact schema drift"
            )
        if (
            any(
                envelope.get(key) != value
                for key, value in expected_identity.items()
            )
            or envelope["lock_canonical_sha256"] != lock_canonical_sha256
            or envelope["cache_spec_cell_id"]
            != expected_cache_spec_cell_id
            or (
                expected_build_receipt_sha256 is not None
                and envelope["authority_lock_build_receipt_sha256"]
                != expected_build_receipt_sha256
            )
        ):
            raise authority.SomphLineageAuthorityError(
                "signed authority envelope pinned identity/binding drift"
            )
        authority._verify_ed25519(
            public_key,
            authority._authority_signature_message(envelope),
            bytes.fromhex(envelope["signature_ed25519_hex"]),
        )

    return verify_test_envelope


def _lock(tmp_path: Path) -> tuple[Path, dict]:
    sha = "a" * 64
    file_descriptor = {
        "path": str(tmp_path / "input.bin"),
        "sha256": sha,
        "size_bytes": 1,
    }
    payload = {
        "schema": authority.AUTHORITY_LOCK_SCHEMA,
        "receiver": "20-1",
        "seed": authority.DEVELOPMENT_SEED,
        "cache_scope": "stage2_target_old",
        "old_tx_ids": list(authority.OLD_TX_IDS),
        "new_tx_ids": [],
        "cache_set_manifest": dict(file_descriptor),
        "cache_sha256_by_scenario": {
            scenario: sha for scenario in authority.FORMAL_LEO_WEAK_SCENARIOS
        },
        "exporter": dict(file_descriptor),
        "build_spec": {
            "path": str(tmp_path / "build_spec.json"),
            "file_sha256": sha,
            "canonical_sha256": sha,
            "size_bytes": 1,
        },
        "channel_code_closure": {
            "closure_sha256": sha,
            "members": [
                {
                    "logical_name": logical_name,
                    **file_descriptor,
                }
                for logical_name in authority.CHANNEL_CODE_LOGICAL_MEMBERS
            ],
        },
        "channel_config_sha256_by_scenario": {
            scenario: sha for scenario in authority.FORMAL_LEO_WEAK_SCENARIOS
        },
        **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        "physical_sample_scenario_assignment_policy": (
            authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": {
            scenario: f"{index + 1:x}" * 64
            for index, scenario in enumerate(
                authority.FORMAL_LEO_WEAK_SCENARIOS
            )
        },
        "physical_sample_scenario_assignment_sha256": "4" * 64,
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "post_channel_iq_sha256_root_by_scenario": {
            scenario: sha for scenario in authority.FORMAL_LEO_WEAK_SCENARIOS
        },
        "overlay_ids_sha256_by_scenario": {
            scenario: sha for scenario in authority.FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_role_inputs_root_sha256": sha,
        "datasets": [
            {
                "role": "target_old",
                **file_descriptor,
                "tx_ids": list(authority.OLD_TX_IDS),
            }
        ],
    }
    assert set(payload) == authority._LOCK_KEYS
    path = tmp_path / "authority_lock.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path, payload


def _registered_lock(tmp_path: Path) -> tuple[Path, dict]:
    path, payload = _lock(tmp_path)
    descriptor = {
        "path": str(tmp_path / "new_input.bin"),
        "sha256": "a" * 64,
        "size_bytes": 1,
    }
    payload["cache_scope"] = "stage2_registered"
    payload["new_tx_ids"] = list(authority.NEW_TX_IDS)
    payload["datasets"].append(
        {
            "role": "target_new",
            **descriptor,
            "tx_ids": list(authority.NEW_TX_IDS),
        }
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path, payload


def _production_build_receipt(lock: dict, *, lock_file_sha: str) -> dict:
    lock_canonical_sha = authority.sha256_bytes(
        authority.canonical_json_bytes(lock)
    )
    receiver, seed, roles = authority._validate_lock_formal_identity(lock)
    dataset_root = authority._dataset_root_from_lock(
        lock,
        receiver=receiver,
        seed=seed,
        roles=roles,
    )
    return {
        "schema": signer.lock_builder.AUTHORITY_LOCK_BUILD_RECEIPT_SCHEMA,
        "status": signer.lock_builder.AUTHORITY_LOCK_BUILD_STATUS,
        "cache_spec_manifest_sha256": (
            signer.lock_builder.FORMAL_CACHE_SPEC_MANIFEST_SHA256
        ),
        "cache_spec_manifest_size_bytes": 1,
        "cache_spec_cell_id": "rx_20_1_seed_713101",
        "required_samples_per_tx": 40,
        "receiver": lock["receiver"],
        "seed": lock["seed"],
        "cache_scope": lock["cache_scope"],
        "cache_set_manifest_sha256": lock["cache_set_manifest"]["sha256"],
        "build_spec_file_sha256": lock["build_spec"]["file_sha256"],
        "build_spec_canonical_sha256": (
            lock["build_spec"]["canonical_sha256"]
        ),
        "exporter_sha256": lock["exporter"]["sha256"],
        "channel_code_closure_sha256": (
            lock["channel_code_closure"]["closure_sha256"]
        ),
        "dataset_authority_root_sha256": dataset_root,
        "cache_role_inputs_root_sha256": lock[
            "cache_role_inputs_root_sha256"
        ],
        **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        "physical_sample_scenario_assignment_policy": (
            authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": lock[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": lock[
            "physical_sample_scenario_assignment_sha256"
        ],
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "cache_sha256_by_scenario": lock["cache_sha256_by_scenario"],
        "channel_config_sha256_by_scenario": lock[
            "channel_config_sha256_by_scenario"
        ],
        "post_channel_iq_sha256_root_by_scenario": lock[
            "post_channel_iq_sha256_root_by_scenario"
        ],
        "overlay_ids_sha256_by_scenario": lock[
            "overlay_ids_sha256_by_scenario"
        ],
        "cache_recompute_audits": {
            scenario: {} for scenario in authority.FORMAL_LEO_WEAK_SCENARIOS
        },
        "authority_lock_sha256": lock_file_sha,
        "authority_lock_canonical_sha256": lock_canonical_sha,
        "external_authority_lock_verified": False,
        "formal_launch_authority": False,
    }


def _production_manifest(lock: dict) -> dict:
    return {
        "schema": "cvs.phase2.somph_registered_cache_build_matrix.v2",
        "formal_launch_authority": False,
        "required_samples_per_tx": 40,
        "cells": [
            {
                "cell_id": "rx_20_1_seed_713101",
                "receiver": lock["receiver"],
                "seed": lock["seed"],
                "cache_scope": lock["cache_scope"],
                "required_samples_per_tx": 40,
                "spec_file_sha256": lock["build_spec"]["file_sha256"],
                "spec_canonical_sha256": (
                    lock["build_spec"]["canonical_sha256"]
                ),
            }
        ],
    }


def _sign(
    tmp_path: Path,
    *,
    private_key: Path,
    public_key: bytes,
    lock_path: Path,
) -> tuple[Path, Path, dict]:
    envelope = tmp_path / "signed_authority_envelope.json"
    receipt = tmp_path / "signing_receipt.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    result = signer._sign_authority_lock_impl(
        lock_path,
        private_key_path=private_key,
        openssl_bin=_openssl(),
        envelope_output=envelope,
        receipt_output=receipt,
        verify_signed_envelope=_test_verifier(public_key),
        receipt_public_key_hex=public_key.hex(),
        receipt_public_key_sha256=hashlib.sha256(public_key).hexdigest(),
        build_authority_binding={
            "authority_lock_build_receipt_sha256": "b" * 64,
            "cache_spec_manifest_sha256": "c" * 64,
            "cache_spec_cell_id": "rx_20_1_seed_713101",
        },
        expected_lock_file_sha256=hashlib.sha256(
            lock_path.read_bytes()
        ).hexdigest(),
        expected_lock_canonical_sha256=authority.sha256_bytes(
            authority.canonical_json_bytes(lock_payload)
        ),
    )
    return envelope, receipt, result


def test_signs_exact_envelope_and_minimal_nonsecret_receipt(
    tmp_path: Path,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, lock = _lock(tmp_path)

    envelope_path, receipt_path, result = _sign(
        tmp_path,
        private_key=private_key,
        public_key=public_key,
        lock_path=lock_path,
    )

    envelope_raw = envelope_path.read_bytes()
    envelope = json.loads(envelope_raw.decode("utf-8"))
    lock_canonical_sha = authority.sha256_bytes(
        authority.canonical_json_bytes(lock)
    )
    assert set(envelope) == authority._ENVELOPE_KEYS
    assert envelope["schema"] == authority.AUTHORITY_ENVELOPE_SCHEMA
    assert envelope["domain"] == authority.AUTHORITY_SIGNATURE_DOMAIN
    assert envelope["issuer"] == authority.PINNED_AUTHORITY_ISSUER
    assert envelope["key_id"] == authority.PINNED_AUTHORITY_KEY_ID
    assert envelope["lock_canonical_sha256"] == lock_canonical_sha
    _test_verifier(public_key)(
        envelope,
        lock_canonical_sha256=lock_canonical_sha,
        expected_cache_spec_cell_id="rx_20_1_seed_713101",
        expected_build_receipt_sha256="b" * 64,
    )

    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw.decode("utf-8"))
    assert set(receipt) == signer._SIGNING_RECEIPT_KEYS
    assert receipt["schema"] == signer.SIGNING_RECEIPT_SCHEMA
    assert receipt["formal_launch_authority"] is False
    assert receipt["lock_file_sha256"] == hashlib.sha256(
        lock_path.read_bytes()
    ).hexdigest()
    assert receipt["lock_canonical_sha256"] == lock_canonical_sha
    assert receipt["signed_authority_envelope_sha256"] == hashlib.sha256(
        envelope_raw
    ).hexdigest()
    assert receipt["pinned_authority_public_key_hex"] == public_key.hex()
    assert receipt["pinned_authority_public_key_sha256"] == hashlib.sha256(
        public_key
    ).hexdigest()
    assert receipt["authority_key_id"] == authority.PINNED_AUTHORITY_KEY_ID
    assert receipt["openssl_binary_sha256"] == hashlib.sha256(
        _openssl().read_bytes()
    ).hexdigest()

    serialized_outputs = receipt_raw + json.dumps(
        result,
        sort_keys=True,
    ).encode("utf-8")
    private_bytes = private_key.read_bytes()
    assert str(private_key.resolve()).encode("utf-8") not in serialized_outputs
    assert hashlib.sha256(private_bytes).hexdigest().encode() not in serialized_outputs
    assert private_bytes not in serialized_outputs
    assert b"private" not in receipt_raw.lower()


def test_wrong_private_key_fails_before_outputs_are_created(
    tmp_path: Path,
) -> None:
    _pinned_private, pinned_public = _generate_key(tmp_path, "pinned")
    wrong_private, _wrong_public = _generate_key(tmp_path, "wrong")
    lock_path, _lock_payload = _lock(tmp_path)

    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="Ed25519 authority signature",
    ) as exc:
        _sign(
            tmp_path,
            private_key=wrong_private,
            public_key=pinned_public,
            lock_path=lock_path,
        )
    assert str(wrong_private.resolve()) not in str(exc.value)
    assert not (tmp_path / "signed_authority_envelope.json").exists()
    assert not (tmp_path / "signing_receipt.json").exists()


def test_production_verifier_rejects_generated_untrusted_key(
    tmp_path: Path,
) -> None:
    private_key, _public_key = _generate_key(tmp_path, "untrusted")
    lock_path, lock_payload = _lock(tmp_path)

    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="Ed25519 authority signature",
    ):
        signer._sign_authority_lock_impl(
            lock_path,
            private_key_path=private_key,
            openssl_bin=_openssl(),
            envelope_output=tmp_path / "signed_authority_envelope.json",
            receipt_output=tmp_path / "signing_receipt.json",
            verify_signed_envelope=authority._verify_signed_envelope,
            receipt_public_key_hex=authority.PINNED_AUTHORITY_PUBLIC_KEY_HEX,
            receipt_public_key_sha256=(
                authority.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
            ),
            build_authority_binding={
                "authority_lock_build_receipt_sha256": "b" * 64,
                "cache_spec_manifest_sha256": (
                    "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
                ),
                "cache_spec_cell_id": "rx_20_1_seed_713101",
            },
            expected_lock_file_sha256=hashlib.sha256(
                lock_path.read_bytes()
            ).hexdigest(),
            expected_lock_canonical_sha256=authority.sha256_bytes(
                authority.canonical_json_bytes(lock_payload)
            ),
        )
    assert not (tmp_path / "signed_authority_envelope.json").exists()
    assert not (tmp_path / "signing_receipt.json").exists()


def test_rejects_nonformal_lock_identity_before_signing(
    tmp_path: Path,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, lock = _lock(tmp_path)
    lock["receiver"] = "not-formal"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="receiver is not formal",
    ):
        _sign(
            tmp_path,
            private_key=private_key,
            public_key=public_key,
            lock_path=lock_path,
        )
    assert not (tmp_path / "signed_authority_envelope.json").exists()
    assert not (tmp_path / "signing_receipt.json").exists()


def test_rejects_v1_lock_and_v1_build_receipt(tmp_path: Path) -> None:
    _legacy_path, legacy_lock = _lock(tmp_path)
    legacy_lock["schema"] = "cvs.phase2.somph_leo_weak_authority_lock.v1"
    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="schema drift",
    ):
        signer._validate_lock_for_signing(legacy_lock)

    _path, lock = _registered_lock(tmp_path)
    lock_file_sha = "a" * 64
    receipt = _production_build_receipt(
        lock,
        lock_file_sha=lock_file_sha,
    )
    receipt["schema"] = "cvs.phase1.somph_authority_lock_build_receipt.v1"
    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="binding drift",
    ):
        authority._verify_build_authority_binding(
            lock,
            lock_file_sha256=lock_file_sha,
            lock_canonical_sha256=authority.sha256_bytes(
                authority.canonical_json_bytes(lock)
            ),
            build_receipt=receipt,
            build_receipt_sha256="b" * 64,
            cache_spec_manifest=_production_manifest(lock),
            cache_spec_manifest_sha256=(
                signer.lock_builder.FORMAL_CACHE_SPEC_MANIFEST_SHA256
            ),
            cache_spec_manifest_size_bytes=1,
        )


@pytest.mark.parametrize(
    "field",
    tuple(authority.PHASE2_SINGLE_OBSERVATION_CONTRACT),
)
def test_signing_rejects_each_single_observation_field_drift(
    tmp_path: Path,
    field: str,
) -> None:
    _path, lock = _lock(tmp_path)
    expected = authority.PHASE2_SINGLE_OBSERVATION_CONTRACT[field]
    lock[field] = (not expected) if isinstance(expected, bool) else "legacy"
    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="single-observation contract drift",
    ):
        signer._validate_lock_for_signing(lock)


def test_rejects_malformed_descriptor_before_signing(
    tmp_path: Path,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, lock = _lock(tmp_path)
    lock["build_spec"]["canonical_sha256"] = "not-a-sha"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="build_spec.canonical_sha256",
    ):
        _sign(
            tmp_path,
            private_key=private_key,
            public_key=public_key,
            lock_path=lock_path,
        )
    assert not (tmp_path / "signed_authority_envelope.json").exists()
    assert not (tmp_path / "signing_receipt.json").exists()


def test_rejects_lock_substitution_after_production_preflight(
    tmp_path: Path,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, original_lock = _lock(tmp_path)
    expected_file_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    expected_canonical_sha = authority.sha256_bytes(
        authority.canonical_json_bytes(original_lock)
    )
    substituted_lock = json.loads(json.dumps(original_lock))
    substituted_lock["cache_sha256_by_scenario"]["leo_clear_weak"] = "d" * 64
    lock_path.write_text(
        json.dumps(substituted_lock, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        signer.SomphAuthoritySigningError,
        match="changed after production preflight",
    ):
        signer._sign_authority_lock_impl(
            lock_path,
            private_key_path=private_key,
            openssl_bin=_openssl(),
            envelope_output=tmp_path / "signed_authority_envelope.json",
            receipt_output=tmp_path / "signing_receipt.json",
            verify_signed_envelope=_test_verifier(public_key),
            receipt_public_key_hex=public_key.hex(),
            receipt_public_key_sha256=hashlib.sha256(public_key).hexdigest(),
            build_authority_binding={
                "authority_lock_build_receipt_sha256": "b" * 64,
                "cache_spec_manifest_sha256": "c" * 64,
                "cache_spec_cell_id": "rx_20_1_seed_713101",
            },
            expected_lock_file_sha256=expected_file_sha,
            expected_lock_canonical_sha256=expected_canonical_sha,
        )
    assert not (tmp_path / "signed_authority_envelope.json").exists()
    assert not (tmp_path / "signing_receipt.json").exists()


def test_production_entry_binds_official_builder_receipt_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock = _registered_lock(tmp_path)
    lock_raw = lock_path.read_bytes()
    lock_sha = hashlib.sha256(lock_raw).hexdigest()
    receipt_path = tmp_path / "authority_lock_build_receipt.json"
    manifest_path = tmp_path / "manifest.json"
    receipt = _production_build_receipt(lock, lock_file_sha=lock_sha)
    manifest = _production_manifest(lock)
    receipt_sha = "b" * 64
    manifest_sha = signer.lock_builder.FORMAL_CACHE_SPEC_MANIFEST_SHA256

    def read_external_json(path: str | Path, *, context: str):
        resolved = Path(path)
        if resolved == lock_path:
            return lock, lock_raw, lock_sha, len(lock_raw)
        if resolved == receipt_path:
            return receipt, b"receipt", receipt_sha, 7
        if resolved == manifest_path:
            return manifest, b"manifest", manifest_sha, 8
        raise AssertionError(f"unexpected read: {context}: {resolved}")

    captured: dict = {}

    def fake_sign_impl(path: str | Path, **kwargs):
        captured["path"] = Path(path)
        captured.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(authority, "_read_external_json", read_external_json)
    monkeypatch.setattr(signer, "_sign_authority_lock_impl", fake_sign_impl)
    result = signer.sign_authority_lock(
        lock_path,
        private_key_path=tmp_path / "private.pem",
        openssl_bin=_openssl(),
        envelope_output=tmp_path / "envelope.json",
        receipt_output=tmp_path / "signing_receipt.json",
        lock_build_receipt_path=receipt_path,
        cache_spec_manifest_path=manifest_path,
    )

    assert result == {"status": "captured"}
    assert captured["path"] == lock_path
    assert captured["expected_lock_file_sha256"] == lock_sha
    assert captured["expected_lock_canonical_sha256"] == (
        receipt["authority_lock_canonical_sha256"]
    )
    assert captured["build_authority_binding"] == {
        "authority_lock_build_receipt_sha256": receipt_sha,
        "cache_spec_manifest_sha256": manifest_sha,
        "cache_spec_cell_id": "rx_20_1_seed_713101",
    }


def test_bundle_validator_rechecks_official_manifest_cell_and_receipt_roots(
    tmp_path: Path,
) -> None:
    lock_path, lock = _registered_lock(tmp_path)
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    lock_canonical_sha = authority.sha256_bytes(
        authority.canonical_json_bytes(lock)
    )
    receipt = _production_build_receipt(lock, lock_file_sha=lock_sha)
    manifest = _production_manifest(lock)

    receipt_sha, dataset_root = authority._verify_build_authority_binding(
        lock,
        lock_file_sha256=lock_sha,
        lock_canonical_sha256=lock_canonical_sha,
        build_receipt=receipt,
        build_receipt_sha256="b" * 64,
        cache_spec_manifest=manifest,
        cache_spec_manifest_sha256=(
            signer.lock_builder.FORMAL_CACHE_SPEC_MANIFEST_SHA256
        ),
        cache_spec_manifest_size_bytes=1,
    )

    assert receipt_sha == "b" * 64
    assert dataset_root == receipt["dataset_authority_root_sha256"]

    tampered = json.loads(json.dumps(receipt))
    tampered["physical_sample_scenario_assignment_sha256"] = "d" * 64
    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="binding drift",
    ):
        authority._verify_build_authority_binding(
            lock,
            lock_file_sha256=lock_sha,
            lock_canonical_sha256=lock_canonical_sha,
            build_receipt=tampered,
            build_receipt_sha256="b" * 64,
            cache_spec_manifest=manifest,
            cache_spec_manifest_sha256=(
                signer.lock_builder.FORMAL_CACHE_SPEC_MANIFEST_SHA256
            ),
            cache_spec_manifest_size_bytes=1,
        )


def test_production_entry_executes_real_signing_with_preflight_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "production")
    lock_path, lock = _registered_lock(tmp_path)
    lock_raw = lock_path.read_bytes()
    lock_sha = hashlib.sha256(lock_raw).hexdigest()
    receipt_path = tmp_path / "authority_lock_build_receipt.json"
    manifest_path = tmp_path / "manifest.json"
    receipt = _production_build_receipt(lock, lock_file_sha=lock_sha)
    manifest = _production_manifest(lock)
    receipt_sha = "b" * 64
    manifest_sha = signer.lock_builder.FORMAL_CACHE_SPEC_MANIFEST_SHA256

    def read_external_json(path: str | Path, *, context: str):
        resolved = Path(path)
        if resolved == lock_path:
            return lock, lock_raw, lock_sha, len(lock_raw)
        if resolved == receipt_path:
            return receipt, b"receipt", receipt_sha, 7
        if resolved == manifest_path:
            return manifest, b"manifest", manifest_sha, 8
        raise AssertionError(f"unexpected read: {context}: {resolved}")

    monkeypatch.setattr(authority, "_read_external_json", read_external_json)
    monkeypatch.setattr(
        authority,
        "_verify_signed_envelope",
        _test_verifier(public_key),
    )
    monkeypatch.setattr(
        authority,
        "PINNED_AUTHORITY_PUBLIC_KEY_HEX",
        public_key.hex(),
    )
    monkeypatch.setattr(
        authority,
        "PINNED_AUTHORITY_PUBLIC_KEY_SHA256",
        hashlib.sha256(public_key).hexdigest(),
    )
    envelope_path = tmp_path / "envelope.json"
    signing_receipt_path = tmp_path / "signing_receipt.json"
    signer.sign_authority_lock(
        lock_path,
        private_key_path=private_key,
        openssl_bin=_openssl(),
        envelope_output=envelope_path,
        receipt_output=signing_receipt_path,
        lock_build_receipt_path=receipt_path,
        cache_spec_manifest_path=manifest_path,
    )

    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["authority_lock_build_receipt_sha256"] == receipt_sha
    assert envelope["cache_spec_manifest_sha256"] == manifest_sha
    assert envelope["cache_spec_cell_id"] == "rx_20_1_seed_713101"
    _test_verifier(public_key)(
        envelope,
        lock_canonical_sha256=receipt[
            "authority_lock_canonical_sha256"
        ],
        expected_cache_spec_cell_id="rx_20_1_seed_713101",
        expected_build_receipt_sha256=receipt_sha,
    )
    signing_receipt = json.loads(
        signing_receipt_path.read_text(encoding="utf-8")
    )
    assert signing_receipt["authority_lock_build_receipt_sha256"] == receipt_sha
    assert signing_receipt["cache_spec_manifest_sha256"] == manifest_sha
    assert signing_receipt["cache_spec_cell_id"] == "rx_20_1_seed_713101"


def test_production_entry_rejects_nonofficial_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock = _registered_lock(tmp_path)
    lock_raw = lock_path.read_bytes()
    lock_sha = hashlib.sha256(lock_raw).hexdigest()
    receipt_path = tmp_path / "authority_lock_build_receipt.json"
    manifest_path = tmp_path / "manifest.json"
    receipt = _production_build_receipt(lock, lock_file_sha=lock_sha)
    manifest = _production_manifest(lock)

    def read_external_json(path: str | Path, *, context: str):
        resolved = Path(path)
        if resolved == lock_path:
            return lock, lock_raw, lock_sha, len(lock_raw)
        if resolved == receipt_path:
            return receipt, b"receipt", "b" * 64, 7
        if resolved == manifest_path:
            return manifest, b"manifest", "d" * 64, 8
        raise AssertionError(f"unexpected read: {context}: {resolved}")

    monkeypatch.setattr(authority, "_read_external_json", read_external_json)
    with pytest.raises(
        signer.SomphAuthoritySigningError,
        match="build receipt/lock binding drift",
    ):
        signer.sign_authority_lock(
            lock_path,
            private_key_path=tmp_path / "private.pem",
            openssl_bin=_openssl(),
            envelope_output=tmp_path / "envelope.json",
            receipt_output=tmp_path / "signing_receipt.json",
            lock_build_receipt_path=receipt_path,
            cache_spec_manifest_path=manifest_path,
        )


def test_rejects_unpinned_openssl_path_even_when_binary_bytes_match(
    tmp_path: Path,
) -> None:
    private_key, _public_key = _generate_key(tmp_path, "authority")
    lock_path, lock_payload = _lock(tmp_path)
    copied_openssl = tmp_path / "copied-openssl.exe"
    shutil.copyfile(_openssl(), copied_openssl)

    with pytest.raises(
        signer.SomphAuthoritySigningError,
        match="path is not pinned",
    ):
        signer._sign_authority_lock_impl(
            lock_path,
            private_key_path=private_key,
            openssl_bin=copied_openssl,
            envelope_output=tmp_path / "signed_authority_envelope.json",
            receipt_output=tmp_path / "signing_receipt.json",
            verify_signed_envelope=authority._verify_signed_envelope,
            receipt_public_key_hex=authority.PINNED_AUTHORITY_PUBLIC_KEY_HEX,
            receipt_public_key_sha256=(
                authority.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
            ),
            build_authority_binding={
                "authority_lock_build_receipt_sha256": "b" * 64,
                "cache_spec_manifest_sha256": "c" * 64,
                "cache_spec_cell_id": "rx_20_1_seed_713101",
            },
            expected_lock_file_sha256=hashlib.sha256(
                lock_path.read_bytes()
            ).hexdigest(),
            expected_lock_canonical_sha256=authority.sha256_bytes(
                authority.canonical_json_bytes(lock_payload)
            ),
        )


def test_signing_executes_locked_private_verified_openssl_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, _lock_payload = _lock(tmp_path)
    real_run = signer.subprocess.run
    real_openssl_sha256 = signer._openssl_sha256
    executed_paths: list[Path] = []
    private_hash_checks: list[Path] = []

    def capture_private_hash(path: Path) -> str:
        if path.parent.name.startswith("somph-authority-openssl-"):
            private_hash_checks.append(path)
        return real_openssl_sha256(path)

    def capture_private_execution(args, *positional, **keywords):
        executable = Path(args[0])
        if len(args) > 1 and args[1] == "pkeyutl":
            executed_paths.append(executable)
            assert executable != _openssl()
            assert executable.parent.name.startswith(
                "somph-authority-openssl-"
            )
            assert hashlib.sha256(executable.read_bytes()).hexdigest() == (
                signer.PINNED_OPENSSL_BINARY_SHA256
            )
            for name, expected_sha in (
                signer._PINNED_OPENSSL_RUNTIME_SHA256.items()
            ):
                runtime = executable.parent / name
                assert hashlib.sha256(runtime.read_bytes()).hexdigest() == (
                    expected_sha
                )
            if os.name == "nt":
                with pytest.raises(OSError):
                    executable.write_bytes(b"malicious")
        return real_run(args, *positional, **keywords)

    monkeypatch.setattr(signer, "_openssl_sha256", capture_private_hash)
    monkeypatch.setattr(signer.subprocess, "run", capture_private_execution)
    _sign(
        tmp_path,
        private_key=private_key,
        public_key=public_key,
        lock_path=lock_path,
    )

    assert len(executed_paths) == 1
    expected_names = {
        "openssl.exe",
        *signer._PINNED_OPENSSL_RUNTIME_SHA256,
    }
    assert {path.name for path in private_hash_checks} == expected_names
    assert all(
        sum(path.name == name for path in private_hash_checks) == 2
        for name in expected_names
    )
    assert not executed_paths[0].exists()
    assert not executed_paths[0].parent.exists()


def test_original_openssl_a_to_malicious_to_a_swap_cannot_reach_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, _lock_payload = _lock(tmp_path)
    source = tmp_path / "attacked-openssl.exe"
    verified_bytes = _openssl().read_bytes()
    verified_sha = hashlib.sha256(verified_bytes).hexdigest()
    source.write_bytes(verified_bytes)
    runtime_files = {
        name: (
            (_openssl().parent / name).read_bytes(),
            expected_sha,
        )
        for name, expected_sha in (
            signer._PINNED_OPENSSL_RUNTIME_SHA256.items()
        )
    }

    monkeypatch.setattr(
        signer,
        "_pinned_openssl_binary",
        lambda _requested: (
            source,
            verified_bytes,
            verified_sha,
            runtime_files,
        ),
    )
    real_run = signer.subprocess.run
    executed_paths: list[Path] = []

    def swap_original_around_exec(args, *positional, **keywords):
        if len(args) > 1 and args[1] == "pkeyutl":
            executed_paths.append(Path(args[0]))
            source.write_bytes(b"malicious")
            try:
                return real_run(args, *positional, **keywords)
            finally:
                source.write_bytes(verified_bytes)
        return real_run(args, *positional, **keywords)

    monkeypatch.setattr(
        signer.subprocess,
        "run",
        swap_original_around_exec,
    )
    _envelope, receipt_path, _result = _sign(
        tmp_path,
        private_key=private_key,
        public_key=public_key,
        lock_path=lock_path,
    )

    assert len(executed_paths) == 1
    assert executed_paths[0] != source
    assert source.read_bytes() == verified_bytes
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["openssl_binary_sha256"] == verified_sha


def test_tampered_envelope_or_lock_binding_is_rejected(
    tmp_path: Path,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, lock = _lock(tmp_path)
    envelope_path, _receipt_path, _result = _sign(
        tmp_path,
        private_key=private_key,
        public_key=public_key,
        lock_path=lock_path,
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    signature = envelope["signature_ed25519_hex"]
    envelope["signature_ed25519_hex"] = (
        signature[:-1] + ("0" if signature[-1] != "0" else "1")
    )
    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="Ed25519 authority signature",
    ):
        _test_verifier(public_key)(
            envelope,
            lock_canonical_sha256=authority.sha256_bytes(
                authority.canonical_json_bytes(lock)
            ),
            expected_cache_spec_cell_id="rx_20_1_seed_713101",
            expected_build_receipt_sha256="b" * 64,
        )

    untampered = json.loads(envelope_path.read_text(encoding="utf-8"))
    lock["receiver"] = "tampered"
    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="pinned identity/binding",
    ):
        _test_verifier(public_key)(
            untampered,
            lock_canonical_sha256=authority.sha256_bytes(
                authority.canonical_json_bytes(lock)
            ),
            expected_cache_spec_cell_id="rx_20_1_seed_713101",
            expected_build_receipt_sha256="b" * 64,
        )


def test_refuses_overwrite_and_preserves_existing_outputs(
    tmp_path: Path,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, _lock_payload = _lock(tmp_path)
    envelope_path, receipt_path, _result = _sign(
        tmp_path,
        private_key=private_key,
        public_key=public_key,
        lock_path=lock_path,
    )
    envelope_before = envelope_path.read_bytes()
    receipt_before = receipt_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _sign(
            tmp_path,
            private_key=private_key,
            public_key=public_key,
            lock_path=lock_path,
        )

    assert envelope_path.read_bytes() == envelope_before
    assert receipt_path.read_bytes() == receipt_before


def test_rolls_back_envelope_when_receipt_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key, public_key = _generate_key(tmp_path, "authority")
    lock_path, _lock_payload = _lock(tmp_path)
    real_writer = signer._write_new_readonly
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic receipt write failure")
        real_writer(path, payload)

    monkeypatch.setattr(signer, "_write_new_readonly", fail_second)
    with pytest.raises(OSError, match="synthetic receipt write failure"):
        _sign(
            tmp_path,
            private_key=private_key,
            public_key=public_key,
            lock_path=lock_path,
        )
    assert not (tmp_path / "signed_authority_envelope.json").exists()
    assert not (tmp_path / "signing_receipt.json").exists()


def test_openssl_subprocess_environment_excludes_injection_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = (
        "OPENSSL_CONF",
        "OPENSSL_MODULES",
        "OPENSSL_ENGINES",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    )
    for key in forbidden:
        monkeypatch.setenv(key, "untrusted")
    environment = signer._clean_openssl_environment()
    assert all(key not in environment for key in forbidden)
    assert environment["LC_ALL"] == "C"


def test_cli_requires_external_private_key_and_openssl_paths() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "sign_cvs_somph_authority_lock.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--authority-lock",
        "--private-key",
        "--openssl-bin",
        "--envelope-output",
        "--receipt-output",
        "--lock-build-receipt",
        "--cache-spec-manifest",
    ):
        assert option in completed.stdout
