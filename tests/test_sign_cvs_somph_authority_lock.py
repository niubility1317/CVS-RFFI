from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cvsrffi import somph_lineage_authority as authority
from scripts import sign_cvs_somph_authority_lock as signer


def _openssl() -> Path:
    executable = shutil.which("openssl")
    if executable is None:
        pytest.skip("OpenSSL is required for authority signing tests")
    return Path(executable).resolve()


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
        "physical_sample_ids_sha256": sha,
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


def _sign(
    tmp_path: Path,
    *,
    private_key: Path,
    public_key: bytes,
    lock_path: Path,
) -> tuple[Path, Path, dict]:
    envelope = tmp_path / "signed_authority_envelope.json"
    receipt = tmp_path / "signing_receipt.json"
    result = signer._sign_authority_lock_impl(
        lock_path,
        private_key_path=private_key,
        openssl_bin=_openssl(),
        envelope_output=envelope,
        receipt_output=receipt,
        verify_signed_envelope=_test_verifier(public_key),
        receipt_public_key_hex=public_key.hex(),
        receipt_public_key_sha256=hashlib.sha256(public_key).hexdigest(),
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
    lock_path, _lock_payload = _lock(tmp_path)

    with pytest.raises(
        authority.SomphLineageAuthorityError,
        match="Ed25519 authority signature",
    ):
        signer.sign_authority_lock(
            lock_path,
            private_key_path=private_key,
            openssl_bin=_openssl(),
            envelope_output=tmp_path / "signed_authority_envelope.json",
            receipt_output=tmp_path / "signing_receipt.json",
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


def test_rejects_unpinned_openssl_path_even_when_binary_bytes_match(
    tmp_path: Path,
) -> None:
    private_key, _public_key = _generate_key(tmp_path, "authority")
    lock_path, _lock_payload = _lock(tmp_path)
    copied_openssl = tmp_path / "copied-openssl.exe"
    shutil.copyfile(_openssl(), copied_openssl)

    with pytest.raises(
        signer.SomphAuthoritySigningError,
        match="path is not pinned",
    ):
        signer.sign_authority_lock(
            lock_path,
            private_key_path=private_key,
            openssl_bin=copied_openssl,
            envelope_output=tmp_path / "signed_authority_envelope.json",
            receipt_output=tmp_path / "signing_receipt.json",
        )


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
    ):
        assert option in completed.stdout
