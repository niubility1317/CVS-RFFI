from __future__ import annotations

import hashlib
import json
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import build_full_ablation_phase1_deployment_bundle as deployment_builder
from cvsrffi import somph_runtime_trust as runtime_trust
from cvsrffi.phase1_adv3b02_deployment_bundle import (
    SIGNATURE_DOMAIN,
    SIGNATURE_ENVELOPE_SCHEMA,
    SIGNING_REQUEST_SCHEMA,
    canonical_json_bytes,
    sha256_bytes,
)
from scripts.build_full_ablation_phase1_deployment_bundle import (
    FullAblationDeploymentError,
    _class_binding_source,
    _completion_receipt,
    _runtime_and_parity,
    prepare,
    sign,
)


def test_class_binding_reuses_handles_but_requires_exact_phase1_tx_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "binding.json"
    payload = {
        "schema": "cvs.phase2.d19_adv3b02_class_binding.v1",
        "checkpoint_sha256": "0" * 64,
        "entries": [
            {
                "class_index": 0,
                "phase1_tx": "tx-a",
                "registered_class_handle": "cls-a",
            },
            {
                "class_index": 1,
                "phase1_tx": "tx-b",
                "registered_class_handle": "cls-b",
            },
        ],
        "evidence": {},
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert _class_binding_source(
        source, expected_phase1_txs=("tx-a", "tx-b")
    ) == ("cls-a", "cls-b")

    payload["entries"].reverse()
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FullAblationDeploymentError, match="TX/order"):
        _class_binding_source(
            source, expected_phase1_txs=("tx-a", "tx-b")
        )


def test_completion_receipt_binds_checkpoint_and_both_original_prototypes(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "completion.json"
    receipt = {
        "ablation_id": "P1-FULL",
        "phase1_training_complete": True,
        "terminal_status": "COMPLETE",
        "exit_code": 0,
        "selected_checkpoint_sha256": "a" * 64,
        "prototype_hashes": {
            "prototype_path": "b" * 64,
            "prototype_json_path": "c" * 64,
        },
        "row_key": "P1-FULL__train_seed_1",
        "run_id": "run-1",
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    loaded = _completion_receipt(
        receipt_path,
        checkpoint_sha256="a" * 64,
        original_prototype_pt_sha256="b" * 64,
        original_prototype_json_sha256="c" * 64,
    )
    assert loaded["row_key"] == "P1-FULL__train_seed_1"

    receipt["prototype_hashes"]["prototype_json_path"] = "d" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(
        FullAblationDeploymentError, match="completion receipt"
    ):
        _completion_receipt(
            receipt_path,
            checkpoint_sha256="a" * 64,
            original_prototype_pt_sha256="b" * 64,
            original_prototype_json_sha256="c" * 64,
        )


def test_formal_runtime_parity_rejects_cpu_before_checkpoint_load(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FullAblationDeploymentError,
        match="requires an available CUDA device",
    ):
        _runtime_and_parity(
            {},
            input_len=256,
            device=torch.device("cpu"),
            runtime_path=tmp_path / "runtime.pt",
            parity_seed=7281105,
            parity_rows=8,
        )


def test_formal_runtime_parity_requires_process_start_cublas_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(
        FullAblationDeploymentError,
        match="before process start",
    ):
        _runtime_and_parity(
            {},
            input_len=256,
            device=torch.device("cuda:0"),
            runtime_path=tmp_path / "runtime.pt",
            parity_seed=7281105,
            parity_rows=8,
        )


@pytest.mark.parametrize(
    ("device_name", "workspace_config", "error_match"),
    [
        (
            "cpu",
            ":4096:8",
            "requires an available CUDA device",
        ),
        (
            "cuda:0",
            None,
            "before process start",
        ),
        (
            "cuda:0",
            ":16:8",
            "before process start",
        ),
    ],
)
def test_formal_prepare_rejects_invalid_cuda_process_before_input_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    workspace_config: str | None,
    error_match: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if workspace_config is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace_config)

    accessed: list[str] = []

    def fail_if_checkpoint_is_accessed(*_args: object, **_kwargs: object) -> None:
        accessed.append("checkpoint")
        raise AssertionError("formal process preflight must run before input access")

    monkeypatch.setattr(
        deployment_builder,
        "sha256_file",
        fail_if_checkpoint_is_accessed,
    )
    monkeypatch.setattr(
        deployment_builder,
        "_load_checkpoint",
        fail_if_checkpoint_is_accessed,
    )
    args = SimpleNamespace(
        device=device_name,
        output_root=tmp_path / "unused-output",
        checkpoint=tmp_path / "unused-checkpoint.pth",
    )
    with pytest.raises(FullAblationDeploymentError, match=error_match):
        prepare(args)
    assert accessed == []


def test_sign_uses_only_pinned_matching_ed25519_key_and_detached_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import sign_cvs_somph_authority_lock as signer

    private_path = tmp_path / "authority.pem"
    private_path.write_bytes(b"private-key-stays-local")
    unsigned = {
        "schema": SIGNATURE_ENVELOPE_SCHEMA,
        "domain": SIGNATURE_DOMAIN,
        "issuer": runtime_trust.PINNED_AUTHORITY_ISSUER,
        "key_id": runtime_trust.PINNED_AUTHORITY_KEY_ID,
        "detached_seal_sha256": "a" * 64,
    }
    message = (
        SIGNATURE_DOMAIN.encode("ascii")
        + b"\x00"
        + canonical_json_bytes(unsigned)
    )
    request_path = tmp_path / "request.json"
    request_path.write_bytes(
        canonical_json_bytes(
            {
                "schema": SIGNING_REQUEST_SCHEMA,
                "signature_message_sha256": sha256_bytes(message),
                "unsigned_signature_envelope": unsigned,
                "outer_content_root_sha256": "b" * 64,
            }
        )
        + b"\n"
    )
    observed = {}

    @contextlib.contextmanager
    def fake_private_openssl(**_kwargs):
        yield tmp_path / "private-openssl.exe"

    def fake_sign(*, openssl_binary, private_key, message):
        observed.update(
            openssl_binary=openssl_binary,
            private_key=private_key,
            message=message,
        )
        return b"\x01" * 64

    def fake_verify(public_key, signed_message, signature):
        observed.update(
            public_key=public_key,
            signed_message=signed_message,
            signature=signature,
        )

    monkeypatch.setattr(
        signer,
        "_pinned_openssl_binary",
        lambda _requested: (
            tmp_path / "openssl.exe",
            b"openssl",
            "c" * 64,
            {},
        ),
    )
    monkeypatch.setattr(
        signer, "_private_openssl_executable", fake_private_openssl
    )
    monkeypatch.setattr(signer, "_sign_with_openssl", fake_sign)
    monkeypatch.setattr(runtime_trust, "verify_ed25519", fake_verify)
    envelope_path = tmp_path / "signature.json"
    receipt = sign(
        SimpleNamespace(
            signing_request=str(request_path),
            authority_private_key=str(private_path),
            openssl_bin="",
            signature_envelope=str(envelope_path),
            sign_receipt="",
        )
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "SIGNED"
    assert set(envelope) == set(unsigned) | {"signature_ed25519_hex"}
    assert str(private_path) not in envelope_path.read_text(encoding="utf-8")
    assert observed["message"] == message
    assert observed["signed_message"] == message
    assert observed["signature"] == b"\x01" * 64


def test_sign_rejects_nonmatching_private_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import sign_cvs_somph_authority_lock as signer

    private_path = tmp_path / "wrong.pem"
    private_path.write_bytes(b"wrong")
    unsigned = {
        "schema": SIGNATURE_ENVELOPE_SCHEMA,
        "domain": SIGNATURE_DOMAIN,
        "issuer": runtime_trust.PINNED_AUTHORITY_ISSUER,
        "key_id": runtime_trust.PINNED_AUTHORITY_KEY_ID,
        "detached_seal_sha256": "a" * 64,
    }
    message = (
        SIGNATURE_DOMAIN.encode("ascii")
        + b"\x00"
        + canonical_json_bytes(unsigned)
    )
    request = tmp_path / "request.json"
    request.write_bytes(
        canonical_json_bytes(
            {
                "schema": SIGNING_REQUEST_SCHEMA,
                "signature_message_sha256": sha256_bytes(message),
                "unsigned_signature_envelope": unsigned,
                "outer_content_root_sha256": "b" * 64,
            }
        )
        + b"\n"
    )

    @contextlib.contextmanager
    def fake_private_openssl(**_kwargs):
        yield tmp_path / "private-openssl.exe"

    monkeypatch.setattr(
        signer,
        "_pinned_openssl_binary",
        lambda _requested: (
            tmp_path / "openssl.exe",
            b"openssl",
            "c" * 64,
            {},
        ),
    )
    monkeypatch.setattr(
        signer, "_private_openssl_executable", fake_private_openssl
    )
    monkeypatch.setattr(
        signer,
        "_sign_with_openssl",
        lambda **_kwargs: b"\x01" * 64,
    )
    monkeypatch.setattr(
        runtime_trust,
        "verify_ed25519",
        lambda *_args: (_ for _ in ()).throw(ValueError("wrong key")),
    )
    with pytest.raises(FullAblationDeploymentError, match="does not match"):
        sign(
            SimpleNamespace(
                signing_request=str(request),
                authority_private_key=str(private_path),
                openssl_bin="",
                signature_envelope=str(tmp_path / "signature.json"),
                sign_receipt="",
            )
        )
