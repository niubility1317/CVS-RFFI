from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_d111_loo_gat_bundle as d111


SOURCE_SEED = bytes(range(32))
OUTER_SEED = bytes(range(32, 64))
SOURCE_SIGNER = "phase1-source-test"
OUTER_SIGNER = "outer-seal-test"


def _ed25519_keypair(seed: bytes) -> tuple[bytes, str]:
    digest = hashlib.sha512(seed).digest()
    scalar_bytes = bytearray(digest[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 63
    scalar_bytes[31] |= 64
    scalar = int.from_bytes(scalar_bytes, "little")
    public_key = d111._ed_encodepoint(d111._ed_scalarmult(d111._ED_B, scalar))
    return digest, public_key.hex()


def _ed25519_sign(seed: bytes, message: bytes) -> str:
    digest, public_key_hex = _ed25519_keypair(seed)
    scalar_bytes = bytearray(digest[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 63
    scalar_bytes[31] |= 64
    scalar = int.from_bytes(scalar_bytes, "little")
    nonce = int.from_bytes(hashlib.sha512(digest[32:] + message).digest(), "little") % d111._ED_L
    encoded_r = d111._ed_encodepoint(d111._ed_scalarmult(d111._ED_B, nonce))
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + bytes.fromhex(public_key_hex) + message).digest(),
        "little",
    ) % d111._ED_L
    encoded_s = ((nonce + challenge * scalar) % d111._ED_L).to_bytes(32, "little")
    return (encoded_r + encoded_s).hex()


@pytest.fixture(autouse=True)
def _trusted_test_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    _source_digest, source_public = _ed25519_keypair(SOURCE_SEED)
    _outer_digest, outer_public = _ed25519_keypair(OUTER_SEED)
    monkeypatch.setattr(
        d111, "_TRUSTED_SOURCE_ED25519_PUBLIC_KEYS", {SOURCE_SIGNER: source_public}
    )
    monkeypatch.setattr(
        d111, "_TRUSTED_OUTER_ED25519_PUBLIC_KEYS", {OUTER_SIGNER: outer_public}
    )


def test_ed25519_verifier_matches_rfc8032_empty_message_vector() -> None:
    public_key = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    signature = (
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    assert d111._verify_ed25519(public_key, b"", signature)
    assert not d111._verify_ed25519(public_key, b"x", signature)


def _inputs(seed: int = 111) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    class_count, domain_count = 6, 8
    q, _ = np.linalg.qr(rng.normal(size=(d111.FEATURE_DIM, d111.RANK)))
    common = q.T
    bases = []
    for _ in range(class_count):
        perturbed, _ = np.linalg.qr(
            common.T + 0.04 * rng.normal(size=(d111.FEATURE_DIM, d111.RANK))
        )
        bases.append(perturbed.T)
    bases = np.asarray(bases)
    anchors = rng.normal(size=(class_count, d111.FEATURE_DIM))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    shifts = rng.normal(scale=0.03, size=(domain_count, d111.RANK))
    centers = (
        anchors[None, :, :]
        + np.einsum("dr,rp->dp", shifts, common)[:, None, :]
        + rng.normal(scale=0.001, size=(domain_count, class_count, d111.FEATURE_DIM))
    )
    centers /= np.linalg.norm(centers, axis=2, keepdims=True)
    radii = 0.08 + rng.uniform(0.0, 0.02, size=(domain_count, class_count))
    return {
        "core": anchors,
        "class_bases": bases,
        "domain_class_centers": centers,
        "class_radii": radii,
        "class_registry": [f"class-{index}" for index in range(class_count)],
        "source_basis_quantization_error_bound": 1.0e-6,
        "source_center_quantization_error_bound": 1.0e-5,
        "checkpoint_sha256": "3" * 64,
        "method_lock_sha256": "4" * 64,
        "generation_code_sha256": "5" * 64,
        "generation_config_sha256": "6" * 64,
    }


def _signed_source_manifest(values: dict[str, object]) -> dict[str, object]:
    content = d111._aggregate_content_sha256(
        core=np.asarray(values["core"]),
        class_bases=np.asarray(values["class_bases"]),
        domain_class_centers=np.asarray(values["domain_class_centers"]),
        class_radii=np.asarray(values["class_radii"]),
        class_registry=values["class_registry"],
        source_basis_quantization_error_bound=float(
            values["source_basis_quantization_error_bound"]
        ),
        source_center_quantization_error_bound=float(
            values["source_center_quantization_error_bound"]
        ),
    )
    manifest = {
        "schema": d111.SOURCE_AGGREGATE_SCHEMA,
        "component_state": d111.SOURCE_AGGREGATE_STATE,
        "formal_phase2_eligible": True,
        "checkpoint_sha256": values["checkpoint_sha256"],
        "signer_id": SOURCE_SIGNER,
        "aggregate_content_sha256": content,
        "aggregate_only": True,
        "generation_stage": "phase1_offline_before_target_access",
    }
    manifest["signature_ed25519_hex"] = _ed25519_sign(
        SOURCE_SEED, d111._canonical_bytes(manifest)
    )
    return manifest


def _build(root: Path, values: dict[str, object] | None = None):
    arguments = dict(_inputs() if values is None else values)
    arguments.setdefault("source_aggregate_manifest", _signed_source_manifest(arguments))
    receipt = d111.build_d111_bundle_from_aggregate(**arguments, output_dir=root)
    signing_payload = d111.d111_outer_signing_payload(root, signer_id=OUTER_SIGNER)
    seal = d111.install_d111_outer_seal(
        root,
        signer_id=OUTER_SIGNER,
        signature_ed25519_hex=_ed25519_sign(OUTER_SEED, signing_payload),
    )
    return {**receipt, **seal}


def _load(root: Path):
    return d111.load_d111_bundle(
        root,
        expected_checkpoint_sha256="3" * 64,
        expected_method_lock_sha256="4" * 64,
        expected_signer_id=OUTER_SIGNER,
    )


def _rewrite_manifest(root: Path, mutate) -> None:
    path = root / d111.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["content_root_sha256"] = d111._content_root(manifest)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = d111._sha256_file(path)
    (root / d111.MANIFEST_SHA_NAME).write_text(
        f"{digest}  {d111.MANIFEST_NAME}\n", encoding="ascii"
    )


def test_formal_build_load_allowlists_readonly_bounds_and_resources(tmp_path: Path) -> None:
    root = tmp_path / "asset"
    receipt = _build(root)
    asset = _load(root)
    assert {item.name for item in root.iterdir()} == {
        d111.NPZ_NAME,
        d111.MANIFEST_NAME,
        d111.MANIFEST_SHA_NAME,
        d111.OUTER_SEAL_NAME,
        d111.OUTER_SEAL_SHA_NAME,
    }
    with np.load(root / d111.NPZ_NAME, allow_pickle=False) as archive:
        assert set(archive.files) == d111.ALLOWED_NPZ_MEMBERS
        for name in archive.files:
            value = archive[name]
            if name not in {"schema", "feature_schema", "class_registry"}:
                assert value.dtype in {np.dtype(np.int8), np.dtype(np.float16)}
    assert asset.anchors.shape == (6, d111.FEATURE_DIM)
    assert asset.basis.shape == (d111.RANK, d111.FEATURE_DIM)
    assert asset.manifest["effective_bundle_state"] == d111.OUTER_SEALED_STATE
    assert asset.manifest["effective_formal_phase2_eligible"] is True
    for array in (asset.anchors, asset.basis, asset.v_g):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        asset.manifest["schema"] = "tampered"
    resource = asset.manifest["resource_receipt"]
    assert receipt["resource_receipt"] == resource
    assert resource["numeric_payload_bytes"] < 4096
    assert resource["persistent_fp32_source_bank_bytes"] == 0
    assert resource["sample_or_query_state_bytes"] == 0
    assert resource["temporary_projection_eigendecomposition_peak_bytes_upper_bound"] > 0
    assert asset.envelope_b >= asset.manifest["envelope_b_unquantized_upper_bound"]
    assert asset.epsilon >= asset.manifest["epsilon_unquantized_upper_bound"]


def test_class_permutation_preserves_shared_projector_and_permutes_records(tmp_path: Path) -> None:
    values = _inputs()
    first_root = tmp_path / "first"
    _build(first_root, values)
    first = _load(first_root)
    permutation = np.asarray([3, 0, 5, 1, 4, 2])
    second_values = dict(values)
    second_values["core"] = np.asarray(values["core"])[permutation]
    second_values["class_bases"] = np.asarray(values["class_bases"])[permutation]
    second_values["domain_class_centers"] = np.asarray(values["domain_class_centers"])[
        :, permutation
    ]
    second_values["class_radii"] = np.asarray(values["class_radii"])[
        :, permutation
    ]
    second_values["class_registry"] = [values["class_registry"][index] for index in permutation]
    second_root = tmp_path / "second"
    _build(second_root, second_values)
    second = _load(second_root)
    np.testing.assert_allclose(
        first.basis.T @ first.basis,
        second.basis.T @ second.basis,
        atol=1.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(first.anchors[permutation], second.anchors, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first.v_g[permutation], second.v_g, atol=0.0, rtol=0.0)
    assert first.envelope_b == second.envelope_b
    assert first.epsilon == second.epsilon


def test_pending_source_degenerate_gap_and_overwrite_fail_closed(tmp_path: Path) -> None:
    pending = _inputs()
    manifest = _signed_source_manifest(pending)
    manifest["formal_phase2_eligible"] = False
    unsigned = {key: value for key, value in manifest.items() if key != "signature_ed25519_hex"}
    manifest["signature_ed25519_hex"] = _ed25519_sign(
        SOURCE_SEED, d111._canonical_bytes(unsigned)
    )
    pending["source_aggregate_manifest"] = manifest
    with pytest.raises(d111.D111BundleError, match="pending or informal"):
        _build(tmp_path / "pending", pending)

    degenerate = _inputs()
    bases = np.zeros((6, d111.RANK, d111.FEATURE_DIM), dtype=np.float64)
    for index in range(6):
        start = index * d111.RANK
        bases[index, :, start : start + d111.RANK] = np.eye(d111.RANK)
    degenerate["class_bases"] = bases
    with pytest.raises(d111.D111BundleError, match="spectral-gap"):
        _build(tmp_path / "gap", degenerate)

    root = tmp_path / "once"
    _build(root)
    with pytest.raises(FileExistsError, match="immutable"):
        _build(root)


def test_source_manifest_binds_actual_aggregate_and_trusted_signature(tmp_path: Path) -> None:
    content_drift = _inputs()
    content_drift["source_aggregate_manifest"] = _signed_source_manifest(content_drift)
    changed_core = np.asarray(content_drift["core"]).copy()
    changed_core[0, 0] += 1.0e-3
    content_drift["core"] = changed_core
    with pytest.raises(d111.D111BundleError, match="manifest binding drift"):
        _build(tmp_path / "content-drift", content_drift)

    signature_drift = _inputs()
    source_manifest = _signed_source_manifest(signature_drift)
    source_manifest["signature_ed25519_hex"] = "a" * 128
    signature_drift["source_aggregate_manifest"] = source_manifest
    with pytest.raises(d111.D111BundleError, match="signature verification"):
        _build(tmp_path / "signature-drift", signature_drift)


def test_release_keyrings_fail_closed_without_enrolled_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_values = _inputs()
    source_values["source_aggregate_manifest"] = _signed_source_manifest(source_values)
    monkeypatch.setattr(d111, "_TRUSTED_SOURCE_ED25519_PUBLIC_KEYS", {})
    with pytest.raises(d111.D111BundleError, match="source aggregate signer is not"):
        _build(tmp_path / "no-source-authority", source_values)

    _source_digest, source_public = _ed25519_keypair(SOURCE_SEED)
    monkeypatch.setattr(
        d111, "_TRUSTED_SOURCE_ED25519_PUBLIC_KEYS", {SOURCE_SIGNER: source_public}
    )
    root = tmp_path / "no-outer-authority"
    arguments = _inputs()
    arguments["source_aggregate_manifest"] = _signed_source_manifest(arguments)
    d111.build_d111_bundle_from_aggregate(**arguments, output_dir=root)
    payload = d111.d111_outer_signing_payload(root, signer_id=OUTER_SIGNER)
    monkeypatch.setattr(d111, "_TRUSTED_OUTER_ED25519_PUBLIC_KEYS", {})
    with pytest.raises(d111.D111BundleError, match="outer signer is not"):
        d111.install_d111_outer_seal(
            root,
            signer_id=OUTER_SIGNER,
            signature_ed25519_hex=_ed25519_sign(OUTER_SEED, payload),
        )


def test_expected_identity_and_signature_drift_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "identity"
    _build(root)
    common = {
        "expected_checkpoint_sha256": "3" * 64,
        "expected_method_lock_sha256": "4" * 64,
        "expected_signer_id": OUTER_SIGNER,
    }
    for field in ("expected_checkpoint_sha256", "expected_method_lock_sha256"):
        arguments = {**common, field: "8" * 64}
        with pytest.raises(d111.D111BundleError, match="drift"):
            d111.load_d111_bundle(root, **arguments)
    with pytest.raises(d111.D111BundleError, match="signer drift"):
        d111.load_d111_bundle(root, **{**common, "expected_signer_id": "wrong"})


def test_directory_manifest_npz_protocol_and_outer_seal_tamper_are_rejected(tmp_path: Path) -> None:
    extra_root = tmp_path / "extra"
    _build(extra_root)
    (extra_root / "query.json").write_text("{}", encoding="utf-8")
    with pytest.raises(d111.D111BundleError, match="directory member"):
        _load(extra_root)

    manifest_root = tmp_path / "manifest"
    _build(manifest_root)
    manifest_path = manifest_root / d111.MANIFEST_NAME
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(d111.D111BundleError, match="manifest SHA"):
        _load(manifest_root)

    protocol_root = tmp_path / "protocol"
    _build(protocol_root)
    _rewrite_manifest(
        protocol_root,
        lambda manifest: manifest["protocol_receipt"].__setitem__("query_access", True),
    )
    with pytest.raises(d111.D111BundleError, match="protocol receipt"):
        _load(protocol_root)

    npz_root = tmp_path / "npz"
    _build(npz_root)
    npz_path = npz_root / d111.NPZ_NAME
    with np.load(npz_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["query_cache"] = np.asarray([1], dtype=np.int8)
    np.savez_compressed(npz_path, **payload)
    _rewrite_manifest(
        npz_root,
        lambda manifest: manifest.__setitem__("component_npz_sha256", d111._sha256_file(npz_path)),
    )
    with pytest.raises(d111.D111BundleError, match="archive member"):
        _load(npz_root)

    outer_root = tmp_path / "outer"
    _build(outer_root)
    _rewrite_manifest(
        outer_root,
        lambda manifest: manifest.__setitem__("generation_code_sha256", "9" * 64),
    )
    with pytest.raises(d111.D111BundleError, match="outer signature payload drift"):
        _load(outer_root)


def test_public_api_has_no_sample_query_truth_score_path_or_experiment_surface() -> None:
    parameters = set(inspect.signature(d111.build_d111_bundle_from_aggregate).parameters)
    forbidden = {
        "sample",
        "samples",
        "physical_id",
        "source_path",
        "query",
        "truth",
        "receiver",
        "role",
        "quota",
    }
    assert not parameters.intersection(forbidden)
    assert set(d111.__all__) == {
        "ALLOWED_NPZ_MEMBERS",
        "D111Bundle",
        "D111BundleError",
        "ENVELOPE_SCHEMA",
        "FEATURE_SCHEMA",
        "NPZ_NAME",
        "SCHEMA",
        "build_d111_bundle_from_aggregate",
        "d111_outer_signing_payload",
        "install_d111_outer_seal",
        "load_d111_bundle",
    }
    assert not any(
        token in name.lower()
        for name in d111.__all__
        for token in ("score", "predict", "g0", "runner", "path", "file", "sha256_file")
    )


def test_manifest_orthogonality_receipt_matches_decoded_basis(tmp_path: Path) -> None:
    root = tmp_path / "asset"
    _build(root)
    asset = _load(root)
    actual = np.linalg.norm(
        asset.basis.astype(np.float64) @ asset.basis.astype(np.float64).T
        - np.eye(d111.RANK),
        ord=2,
    )
    assert actual == pytest.approx(asset.manifest["u_orthogonality_error"], abs=1.0e-7)
    assert asset.manifest["spectral_gap"] > asset.manifest["spectral_gap_required_minimum"]
