from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path

import pytest

import cvsrffi.somph_runtime_trust as somph_runtime_trust
import cvsrffi.stage2_d105_phase1_authority as authority


TEST_ONLY_SEED = bytes.fromhex(
    "4f3c2a19080706050403020100112233445566778899aabbccddeeff10203040"
)
NOW = datetime(2026, 7, 31, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _test_pinned_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a test-only key; production still reads the fixed trust root."""

    public = authority.ed25519_public_key_from_seed(TEST_ONLY_SEED)
    monkeypatch.setattr(somph_runtime_trust, "PINNED_AUTHORITY_PUBLIC_KEY_HEX", public.hex())
    monkeypatch.setattr(
        somph_runtime_trust,
        "PINNED_AUTHORITY_PUBLIC_KEY_SHA256",
        hashlib.sha256(public).hexdigest(),
    )
    monkeypatch.setattr(somph_runtime_trust, "PINNED_AUTHORITY_KEY_ID", "d105-test-authority-ed25519")


def _identity() -> dict[str, str]:
    fields = (
        "component_manifest_sha256",
        "bundle_wire_sha256",
        "bundle_content_root_sha256",
        "bundle_receipt_root_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "strict_tap_receipt_sha256",
        "source_held_gate_receipt_sha256",
        "validated_bundle_id_sha256",
        "validator_receipt_sha256",
    )
    identity = {name: hashlib.sha256(name.encode("ascii")).hexdigest() for name in fields}
    identity["candidate_id"] = "D105-CBRC+LPO-RC"
    return identity


def _revocation() -> tuple[dict[str, object], bytes]:
    manifest = {
        "schema": authority.D102_REVOCATION_SCHEMA,
        "signature_domain": authority.D102_REVOCATION_SIGNATURE_DOMAIN,
        "issuer_key_id": somph_runtime_trust.PINNED_AUTHORITY_KEY_ID,
        "issued_at": "2026-01-01T00:00:00Z",
        "not_before": "2026-01-01T00:00:00Z",
        "expires_at": "2030-01-01T00:00:00Z",
        "revocation_id": "a" * 64,
        "revoked_artifacts": [
            {
                "run_id": "d102_rb_metabias4_phase1_analytic_held_20260724_r6",
                "bundle_manifest_sha256": "0690f2ab19560a54c96599ffc59a56fd31786f48ac2f05659414d8c29ff0da64",
                "bundle_payload_sha256": "440ff82a1f74b67078f699eaca86e85b9739d574721ccfb460a423ff97cc93d4",
                "bundle_seal_sha256": "cdcfceb5a31e3409ccea137fe116347f2214640e6514b080d442e7a193a0db59",
                "bundle_content_root_sha256": "16b9a8388c612509e4b220f2883fcd92187e1de0e4236ef25e2ef72a472a48b7",
                "checkpoint_sha256": "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98",
                "method_lock_sha256": "9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f",
                "runtime_sha256": "e1b21bee74941dfb550b67698a75f485937bc39431ed7859baaa20d44a4899f3",
                "held_score_sha256": "01a45e11fe519389071cf1eb279d293c958fc4fa48e0ed4c51bea9ff20c536b2",
                "tap_archive_sha256": "c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1",
                "status": "PHASE1_HELD_FALSIFIER_REJECT",
            }
        ],
    }
    signature = authority.sign_d105_detached(
        domain=authority.D102_REVOCATION_SIGNATURE_DOMAIN,
        payload=authority.canonical_bytes(manifest),
        private_seed=TEST_ONLY_SEED,
    )
    return manifest, signature


def _review(identity: dict[str, str], *, p0: int = 0, p1: int = 0) -> dict[str, object]:
    return {
        "schema": authority.INDEPENDENT_REVIEW_SCHEMA,
        "candidate_id": identity["candidate_id"],
        "component_manifest_sha256": identity["component_manifest_sha256"],
        "bundle_content_root_sha256": identity["bundle_content_root_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "method_lock_sha256": identity["method_lock_sha256"],
        "d105_candidate_runtime_manifest_sha256": identity[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": identity[
            "d105_candidate_method_lock_sha256"
        ],
        "reviewer_id": "independent-reviewer",
        "reviewed_at": "2026-07-31T00:00:00Z",
        "review_p0": p0,
        "review_p1": p1,
    }


def _signed_envelope(
    identity: dict[str, str],
    *,
    issued_at: str = "2026-07-31T00:00:00Z",
    not_before: str = "2026-07-31T00:00:00Z",
    expires_at: str = "2027-07-31T00:00:00Z",
    nonce: str = "b" * 64,
    nonce_ledger_identity_sha256: str = "f" * 64,
) -> tuple[dict[str, object], bytes, str, str]:
    revocation, revocation_signature = _revocation()
    revocation_sha = authority.validate_d102_revocation_manifest(
        revocation, revocation_signature, now_utc=NOW
    )
    review_sha = authority.validate_independent_review_receipt(
        _review(identity), identity=identity
    )
    envelope = authority.build_d105_authority_envelope(
        identity=identity,
        independent_review_receipt_sha256=review_sha,
        d102_revocation_manifest_sha256=revocation_sha,
        nonce_ledger_identity_sha256=nonce_ledger_identity_sha256,
        issued_at=issued_at,
        not_before=not_before,
        expires_at=expires_at,
        nonce=nonce,
        run_id="d105-test-run-001",
        git_commit="c" * 40,
    )
    signature = authority.sign_d105_detached(
        domain=authority.AUTHORITY_SIGNATURE_DOMAIN,
        payload=authority.canonical_bytes(envelope),
        private_seed=TEST_ONLY_SEED,
    )
    return envelope, signature, review_sha, revocation_sha


def _target25_prepare_binding(
    *, nonce_ledger_identity_sha256: str | None = None
) -> dict[str, object]:
    return {
        "run_id": "d105-target25-prepare-001",
        "git_commit": "c" * 40,
        "matrix_index_sha256": hashlib.sha256(b"matrix-index").hexdigest(),
        "prepare_receipt_file_sha256": hashlib.sha256(
            b"prepare-receipt-file"
        ).hexdigest(),
        "prepare_receipt_sha256": hashlib.sha256(b"prepare-receipt-self").hexdigest(),
        "plan_manifest_sha256": hashlib.sha256(b"plan-manifest").hexdigest(),
        "context_manifest_sha256": hashlib.sha256(b"context-manifest").hexdigest(),
        "plan_receipt_sha256": hashlib.sha256(b"plan-receipt").hexdigest(),
        "authority_envelope_root_sha256": hashlib.sha256(
            b"authority-envelope-root"
        ).hexdigest(),
        "d105_candidate_runtime_manifest_sha256": hashlib.sha256(
            b"candidate-runtime"
        ).hexdigest(),
        "d105_candidate_method_lock_sha256": hashlib.sha256(
            b"candidate-method"
        ).hexdigest(),
        "claim_scope": authority.TARGET25_DEVELOPMENT_CLAIM_SCOPE,
        "formal_launch_authority": False,
        "nonce_ledger_identity_sha256": (
            hashlib.sha256(b"target25-prepare-ledger").hexdigest()
            if nonce_ledger_identity_sha256 is None
            else nonce_ledger_identity_sha256
        ),
    }


def _signed_target25_prepare(
    binding: dict[str, object], *, nonce: str = "e" * 64
) -> tuple[dict[str, object], bytes]:
    envelope = authority.build_d105_target25_prepare_envelope(
        binding=binding,
        issued_at="2026-07-31T00:00:00Z",
        not_before="2026-07-31T00:00:00Z",
        expires_at="2027-07-31T00:00:00Z",
        nonce=nonce,
    )
    signature = authority.sign_d105_detached(
        domain=authority.TARGET25_PREPARE_SIGNATURE_DOMAIN,
        payload=authority.canonical_bytes(envelope),
        private_seed=TEST_ONLY_SEED,
    )
    return envelope, signature


def test_signed_d102_revocation_rejects_real_immutable_identity_even_when_renamed() -> None:
    manifest, signature = _revocation()
    digest = authority.validate_d102_revocation_manifest(
        manifest, signature, now_utc=NOW
    )
    assert digest == hashlib.sha256(authority.canonical_bytes(manifest)).hexdigest()
    with pytest.raises(authority.D105AuthorityError, match="revoked immutable content"):
        authority.reject_revoked_d102_identity(
            manifest,
            run_id="renamed-not-d102",
            bundle_payload_sha256="440ff82a1f74b67078f699eaca86e85b9739d574721ccfb460a423ff97cc93d4",
        )
    tampered = dict(manifest)
    tampered["revocation_id"] = "d" * 64
    with pytest.raises(authority.D105AuthorityError, match="signature"):
        authority.validate_d102_revocation_manifest(tampered, signature, now_utc=NOW)


def test_authority_envelope_rejects_bad_signature_domain_time_replay_and_review() -> None:
    identity = _identity()
    envelope, signature, review_sha, revocation_sha = _signed_envelope(identity)
    guard: set[tuple[str, str]] = set()
    digest = authority.validate_d105_authority_envelope(
        envelope,
        signature,
        identity=identity,
        independent_review_receipt_sha256=review_sha,
        d102_revocation_manifest_sha256=revocation_sha,
        now_utc=NOW,
        nonce_guard=guard,
    )
    assert digest == hashlib.sha256(authority.canonical_bytes(envelope)).hexdigest()
    with pytest.raises(authority.D105AuthorityError, match="nonce replay"):
        authority.validate_d105_authority_envelope(
            envelope,
            signature,
            identity=identity,
            independent_review_receipt_sha256=review_sha,
            d102_revocation_manifest_sha256=revocation_sha,
            now_utc=NOW,
            nonce_guard=guard,
        )

    tampered = dict(envelope)
    tampered["component_manifest_sha256"] = "e" * 64
    with pytest.raises(authority.D105AuthorityError, match="binding drift"):
        authority.validate_d105_authority_envelope(
            tampered,
            signature,
            identity=identity,
            independent_review_receipt_sha256=review_sha,
            d102_revocation_manifest_sha256=revocation_sha,
            now_utc=NOW,
        )

    wrong_domain = dict(envelope)
    wrong_domain["signature_domain"] = authority.D102_REVOCATION_SIGNATURE_DOMAIN
    with pytest.raises(authority.D105AuthorityError, match="domain/schema"):
        authority.validate_d105_authority_envelope(
            wrong_domain,
            signature,
            identity=identity,
            independent_review_receipt_sha256=review_sha,
            d102_revocation_manifest_sha256=revocation_sha,
            now_utc=NOW,
        )

    future, future_signature, _, _ = _signed_envelope(
        identity,
        issued_at="2027-01-01T00:00:00Z",
        not_before="2027-01-01T00:00:00Z",
        expires_at="2028-01-01T00:00:00Z",
        nonce="c" * 64,
    )
    with pytest.raises(authority.D105AuthorityError, match="not yet valid"):
        authority.validate_d105_authority_envelope(
            future,
            future_signature,
            identity=identity,
            independent_review_receipt_sha256=review_sha,
            d102_revocation_manifest_sha256=revocation_sha,
            now_utc=NOW,
        )

    expired, expired_signature, _, _ = _signed_envelope(
        identity,
        issued_at="2025-01-01T00:00:00Z",
        not_before="2025-01-01T00:00:00Z",
        expires_at="2026-01-01T00:00:00Z",
        nonce="d" * 64,
    )
    with pytest.raises(authority.D105AuthorityError, match="expired"):
        authority.validate_d105_authority_envelope(
            expired,
            expired_signature,
            identity=identity,
            independent_review_receipt_sha256=review_sha,
            d102_revocation_manifest_sha256=revocation_sha,
            now_utc=NOW,
        )

    with pytest.raises(authority.D105AuthorityError, match="P0/P1"):
        authority.validate_independent_review_receipt(_review(identity, p0=1), identity=identity)
    with pytest.raises(authority.D105AuthorityError, match="P0/P1"):
        authority.validate_independent_review_receipt(_review(identity, p1=1), identity=identity)


def test_target25_prepare_authority_is_signed_exact_and_development_only(
    tmp_path,
) -> None:
    ledger = tmp_path / "target25-prepare-ledger"
    ledger.mkdir()
    binding = _target25_prepare_binding(
        nonce_ledger_identity_sha256=authority.compute_d105_nonce_ledger_identity(
            ledger,
            run_id="d105-target25-prepare-001",
            signature_domain=authority.TARGET25_PREPARE_SIGNATURE_DOMAIN,
        )
    )
    envelope, signature = _signed_target25_prepare(binding)
    guard: set[tuple[str, str]] = set()
    digest = authority.validate_d105_target25_prepare_envelope(
        envelope,
        signature,
        expected_binding=binding,
        now_utc=NOW,
        nonce_guard=guard,
    )
    assert digest == hashlib.sha256(authority.canonical_bytes(envelope)).hexdigest()
    with pytest.raises(authority.D105AuthorityError, match="nonce replay"):
        authority.validate_d105_target25_prepare_envelope(
            envelope,
            signature,
            expected_binding=binding,
            now_utc=NOW,
            nonce_guard=guard,
        )

    changed_binding = dict(binding)
    changed_binding["prepare_receipt_file_sha256"] = "f" * 64
    with pytest.raises(authority.D105AuthorityError, match="binding drift"):
        authority.validate_d105_target25_prepare_envelope(
            envelope,
            signature,
            expected_binding=changed_binding,
            now_utc=NOW,
        )

    illegal_formal = dict(envelope)
    illegal_formal["formal_launch_authority"] = True
    with pytest.raises(authority.D105AuthorityError, match="deny formal launch"):
        authority.validate_d105_target25_prepare_envelope(
            illegal_formal,
            signature,
            expected_binding=binding,
            now_utc=NOW,
        )

    wrong_domain_signature = authority.sign_d105_detached(
        domain=authority.AUTHORITY_SIGNATURE_DOMAIN,
        payload=authority.canonical_bytes(envelope),
        private_seed=TEST_ONLY_SEED,
    )
    with pytest.raises(authority.D105AuthorityError, match="signature invalid"):
        authority.validate_d105_target25_prepare_envelope(
            envelope,
            wrong_domain_signature,
            expected_binding=binding,
            now_utc=NOW,
        )

    envelope_path = tmp_path / authority.TARGET25_PREPARE_ENVELOPE_NAME
    signature_path = tmp_path / authority.TARGET25_PREPARE_SIGNATURE_NAME
    envelope_path.write_bytes(authority.canonical_bytes(envelope))
    signature_path.write_bytes(signature)
    loaded = authority.load_signed_d105_target25_prepare_envelope(
        envelope_path,
        signature_path,
        expected_binding=binding,
        now_utc=NOW,
    )
    assert loaded["envelope_sha256"] == digest
    alternate_ledger = tmp_path / "target25-prepare-alternate-ledger"
    alternate_ledger.mkdir()
    with pytest.raises(authority.D105AuthorityError, match="ledger identity drift"):
        authority.consume_target25_prepare_nonce_once(
            alternate_ledger,
            envelope=envelope,
            envelope_sha256=digest,
        )
    authority.consume_target25_prepare_nonce_once(
        ledger, envelope=envelope, envelope_sha256=digest
    )
    with pytest.raises(authority.D105AuthorityError, match="nonce replay"):
        authority.consume_target25_prepare_nonce_once(
            ledger, envelope=envelope, envelope_sha256=digest
        )


def test_target25_prepare_signer_cli_recomputes_file_bindings(tmp_path) -> None:
    def write_json(path, value) -> str:
        payload = authority.canonical_bytes(value) + b"\n"
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_bytes(b'{"matrix":"D105-target25"}\n')
    shared_identity = {
        "claim_scope": authority.TARGET25_DEVELOPMENT_CLAIM_SCOPE,
        "formal_launch_authority": False,
        "authority_envelope_root_sha256": hashlib.sha256(
            b"authority-envelope-root"
        ).hexdigest(),
        "data_feature_runtime_sha256": hashlib.sha256(b"data-feature").hexdigest(),
        "data_materialization_lock_sha256": hashlib.sha256(
            b"data-materialization"
        ).hexdigest(),
        "d105_candidate_runtime_manifest_sha256": hashlib.sha256(
            b"candidate-runtime"
        ).hexdigest(),
        "d105_candidate_method_lock_sha256": hashlib.sha256(
            b"candidate-method"
        ).hexdigest(),
    }
    plan_path = tmp_path / "target25_plan.json"
    context_path = tmp_path / "target25_context.json"
    plan_payload = {
        "schema": "cvs.phase2.d105.target25_runner.v1.plan",
        "seed": 713102,
        **shared_identity,
        "arms": ["M0", "M_DA", "M_HEAD", "M_JOINT"],
        "leo_scenarios": [
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        ],
        "target25_slices": [[10, 5], [10, 10], [10, 20], [5, 20], [1, 20]],
        "rows": [],
    }
    plan_receipt_sha = hashlib.sha256(authority.canonical_bytes(plan_payload)).hexdigest()
    plan_without_receipt = {
        "schema": "cvs.phase2.d105.target25_runner.v1.plan_manifest",
        "plan_payload": plan_payload,
        "candidate_identity_sources": {
            "candidate_runtime_manifest_path": "target25_plan.json.inputs/runtime.json",
            "candidate_method_lock_path": "target25_plan.json.inputs/method.json",
        },
        "plan_receipt_sha256": plan_receipt_sha,
    }
    plan = {
        **plan_without_receipt,
        "plan_manifest_receipt_sha256": hashlib.sha256(
            authority.canonical_bytes(plan_without_receipt)
        ).hexdigest(),
    }
    plan_sha = write_json(plan_path, plan)
    context_without_receipt = {
        "schema": "cvs.phase2.d105.target25_context_manifest.v1",
        "plan_receipt_sha256": plan_receipt_sha,
        **shared_identity,
        "rows": [],
    }
    context = {
        **context_without_receipt,
        "context_manifest_receipt_sha256": hashlib.sha256(
            authority.canonical_bytes(context_without_receipt)
        ).hexdigest(),
    }
    context_sha = write_json(context_path, context)
    receipt_unsigned = {
        "schema": "cvs.phase2.d105.target25_prepare_receipt.v1",
        "status": "TARGET25_INPUTS_PREPARED",
        "promotable": False,
        "matrix_index_sha256": hashlib.sha256(matrix_index.read_bytes()).hexdigest(),
        "plan_manifest_sha256": plan_sha,
        "context_manifest_sha256": context_sha,
        "plan_receipt_sha256": plan_receipt_sha,
        **shared_identity,
        "outer_row_count": 25,
    }
    receipt = {
        **receipt_unsigned,
        "prepare_receipt_sha256": hashlib.sha256(
            authority.canonical_bytes(receipt_unsigned)
        ).hexdigest(),
    }
    receipt_path = tmp_path / "prepare_receipt.json"
    write_json(receipt_path, receipt)
    seed_path = tmp_path / "test-authority-seed.bin"
    seed_path.write_bytes(TEST_ONLY_SEED)
    envelope_path = tmp_path / authority.TARGET25_PREPARE_ENVELOPE_NAME
    signature_path = tmp_path / authority.TARGET25_PREPARE_SIGNATURE_NAME
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "sign_d105_phase1_authority.py"
    )
    spec = importlib.util.spec_from_file_location("d105_authority_signer_test", script)
    assert spec is not None and spec.loader is not None
    signer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signer)
    assert signer.main(
        [
            "sign-target25-prepare",
            "--prepare-receipt",
            str(receipt_path),
            "--matrix-index",
            str(matrix_index),
            "--plan-manifest",
            str(plan_path),
            "--context-manifest",
            str(context_path),
            "--run-id",
            "d105-target25-prepare-001",
            "--git-commit",
            "c" * 40,
            "--issued-at",
            "2026-07-31T00:00:00Z",
            "--not-before",
            "2026-07-31T00:00:00Z",
            "--expires-at",
            "2027-07-31T00:00:00Z",
            "--nonce",
            "e" * 64,
            "--nonce-ledger-identity-sha256",
            "f" * 64,
            "--private-seed-file",
            str(seed_path),
            "--output-envelope",
            str(envelope_path),
            "--output-signature",
            str(signature_path),
        ]
    ) == 0
    binding = {
        "run_id": "d105-target25-prepare-001",
        "git_commit": "c" * 40,
        "matrix_index_sha256": receipt_unsigned["matrix_index_sha256"],
        "prepare_receipt_file_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "prepare_receipt_sha256": receipt["prepare_receipt_sha256"],
        "plan_manifest_sha256": plan_sha,
        "context_manifest_sha256": context_sha,
        "plan_receipt_sha256": plan_receipt_sha,
        "claim_scope": shared_identity["claim_scope"],
        "formal_launch_authority": shared_identity["formal_launch_authority"],
        "authority_envelope_root_sha256": shared_identity[
            "authority_envelope_root_sha256"
        ],
        "d105_candidate_runtime_manifest_sha256": shared_identity[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": shared_identity[
            "d105_candidate_method_lock_sha256"
        ],
        "nonce_ledger_identity_sha256": "f" * 64,
    }
    loaded = authority.load_signed_d105_target25_prepare_envelope(
        envelope_path,
        signature_path,
        expected_binding=binding,
        now_utc=NOW,
    )
    assert loaded["envelope"]["prepare_receipt_file_sha256"] == binding[
        "prepare_receipt_file_sha256"
    ]

    counterfeit_plan = {**plan, **shared_identity}
    counterfeit_path = tmp_path / "target25_plan_top_level_counterfeit.json"
    counterfeit_plan_sha = write_json(counterfeit_path, counterfeit_plan)
    counterfeit_receipt_unsigned = {
        **receipt_unsigned,
        "plan_manifest_sha256": counterfeit_plan_sha,
    }
    counterfeit_receipt = {
        **counterfeit_receipt_unsigned,
        "prepare_receipt_sha256": hashlib.sha256(
            authority.canonical_bytes(counterfeit_receipt_unsigned)
        ).hexdigest(),
    }
    counterfeit_receipt_path = tmp_path / "prepare_receipt_counterfeit.json"
    write_json(counterfeit_receipt_path, counterfeit_receipt)
    with pytest.raises(SystemExit):
        signer.main(
            [
                "sign-target25-prepare",
                "--prepare-receipt",
                str(counterfeit_receipt_path),
                "--matrix-index",
                str(matrix_index),
                "--plan-manifest",
                str(counterfeit_path),
                "--context-manifest",
                str(context_path),
                "--run-id",
                "d105-target25-prepare-001",
                "--git-commit",
                "c" * 40,
                "--issued-at",
                "2026-07-31T00:00:00Z",
                "--not-before",
                "2026-07-31T00:00:00Z",
                "--expires-at",
                "2027-07-31T00:00:00Z",
                "--nonce",
                "f" * 64,
                "--nonce-ledger-identity-sha256",
                "f" * 64,
                "--private-seed-file",
                str(seed_path),
                "--output-envelope",
                str(tmp_path / "counterfeit.envelope.json"),
                "--output-signature",
                str(tmp_path / "counterfeit.envelope.ed25519"),
            ]
        )


def test_pinned_key_rejects_signature_from_another_key_and_nonce_ledger_is_one_shot(
    tmp_path,
) -> None:
    identity = _identity()
    ledger = tmp_path / "nonce-ledger"
    ledger.mkdir()
    envelope, signature, review_sha, revocation_sha = _signed_envelope(
        identity,
        nonce_ledger_identity_sha256=authority.compute_d105_nonce_ledger_identity(
            ledger,
            run_id="d105-test-run-001",
            signature_domain=authority.AUTHORITY_SIGNATURE_DOMAIN,
        ),
    )
    with pytest.raises(authority.D105AuthorityError, match="does not match pinned"):
        authority.sign_d105_detached(
            domain=authority.AUTHORITY_SIGNATURE_DOMAIN,
            payload=authority.canonical_bytes(envelope),
            private_seed=b"x" * 32,
        )
    wrong_public = authority.ed25519_public_key_from_seed(b"x" * 32)
    original_key = somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX
    original_sha = somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
    somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX = wrong_public.hex()
    somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256 = hashlib.sha256(wrong_public).hexdigest()
    try:
        with pytest.raises(authority.D105AuthorityError, match="signature invalid"):
            authority.validate_d105_authority_envelope(
                envelope,
                signature,
                identity=identity,
                independent_review_receipt_sha256=review_sha,
                d102_revocation_manifest_sha256=revocation_sha,
                now_utc=NOW,
            )
    finally:
        somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX = original_key
        somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256 = original_sha

    digest = hashlib.sha256(authority.canonical_bytes(envelope)).hexdigest()
    authority.consume_authority_nonce_once(
        ledger, envelope=envelope, envelope_sha256=digest
    )
    with pytest.raises(authority.D105AuthorityError, match="nonce replay"):
        authority.consume_authority_nonce_once(
            ledger, envelope=envelope, envelope_sha256=digest
        )
