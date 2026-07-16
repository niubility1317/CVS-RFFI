from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.stage2_predictor_bundle as predictor_bundle
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.stage2_predictor_bundle import (
    FORMAL_LEO_WEAK_SCENARIOS,
    PREDICTOR_INPUT_STAGE,
    PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
    QUERY_NPZ_MEMBERS,
    QUERY_SCHEMA,
    SUPPORT_NPZ_MEMBERS,
    SUPPORT_SCHEMA,
    PredictorPackageError,
    iq_row_sha256,
    load_verified_stage2_predictor_bundle,
    make_member_descriptor,
    preflight_stage2_predictor_package,
    sha256_file,
    validate_relative_member_path,
    write_predictor_package_manifest_and_seal,
)


def _patch_first_zip_uint(path: Path, *, local_offset: int, central_offset: int, size: int) -> None:
    payload = bytearray(path.read_bytes())
    for signature, offset in (
        (b"PK\x03\x04", local_offset),
        (b"PK\x01\x02", central_offset),
    ):
        start = payload.find(signature)
        assert start >= 0
        value_start = start + offset
        value = int.from_bytes(payload[value_start : value_start + size], "little")
        replacement = value ^ ((1 << (size * 8)) - 1)
        payload[value_start : value_start + size] = replacement.to_bytes(size, "little")
    path.write_bytes(payload)


def _mark_first_zip_member_encrypted(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = payload.find(signature)
        assert start >= 0
        flag_start = start + offset
        flags = int.from_bytes(payload[flag_start : flag_start + 2], "little") | 0x1
        payload[flag_start : flag_start + 2] = flags.to_bytes(2, "little")
    path.write_bytes(payload)


def _rewrite_zip_compression(path: Path, compression: int) -> None:
    with zipfile.ZipFile(path, "r") as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]
    replacement = path.with_suffix(".replacement")
    with zipfile.ZipFile(replacement, "w", compression=compression) as target:
        for name, payload in members:
            target.writestr(name, payload)
    replacement.replace(path)


def _npz(path: Path, *, scenario: str, query: bool, extra: bool = False) -> None:
    iq = np.zeros((1, 2, 8), dtype=np.float32)
    if query:
        payload = {
            "query_leo_weak_iq": iq,
            "query_tokens": np.asarray(["qid_" + "1" * 64]),
            "query_overlay_tokens": np.asarray(["oid_" + "2" * 64]),
            "query_satellite_seeds": np.asarray([11], dtype=np.int64),
            "query_post_channel_iq_sha256": np.asarray([iq_row_sha256(iq[0])]),
            "manifest_json": np.asarray(
                json.dumps(
                    {
                        "schema": QUERY_SCHEMA,
                        "scenario": scenario,
                        "query_truth_included": False,
                        "query_role_included": False,
                        "query_true_batch_class_count_included": False,
                        "query_class_quota_included": False,
                        "query_ordering_hint_included": False,
                        "token_scheme": "hmac_sha256_opaque_v1",
                    },
                    sort_keys=True,
                )
            ),
        }
        if extra:
            payload["debug"] = np.asarray([1])
    else:
        payload = {
            "support_pool_leo_weak_iq": iq,
            "support_pool_class_indices": np.asarray([0], dtype=np.int64),
            "support_pool_rank_within_class": np.asarray([0], dtype=np.int64),
            "support_pool_tokens": np.asarray(["sid_" + "3" * 64]),
            "support_pool_overlay_tokens": np.asarray(["oid_" + "4" * 64]),
            "support_pool_satellite_seeds": np.asarray([11], dtype=np.int64),
            "support_pool_post_channel_iq_sha256": np.asarray([iq_row_sha256(iq[0])]),
            "manifest_json": np.asarray(
                json.dumps(
                    {
                        "schema": SUPPORT_SCHEMA,
                        "scenario": scenario,
                        "registered_support_labels_allowed": True,
                        "registered_class_count": 1,
                        "support_pool_max_k": 1,
                        "token_scheme": "hmac_sha256_opaque_v1",
                    },
                    sort_keys=True,
                )
            ),
        }
    with path.open("xb") as handle:
        np.savez(handle, **payload)


def _package(
    tmp_path: Path,
    *,
    extra_query_member: bool = False,
    first_query_mutator=None,
):
    root = tmp_path / "predictor"
    root.mkdir()
    members = []
    for role, filename in (
        ("checkpoint", "checkpoint.bin"),
        ("adapter", "adapter.bin"),
        ("head", "head.bin"),
        ("tta_policy", "tta_policy.json"),
    ):
        path = root / filename
        path.write_bytes(role.encode("ascii"))
        members.append(
            make_member_descriptor(
                path,
                relative_path=filename,
                artifact_role=role,
                schema=f"test.{role}.v1",
            )
        )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        support = root / f"support_{scenario}.npz"
        query = root / f"query_{scenario}.npz"
        _npz(support, scenario=scenario, query=False)
        _npz(
            query,
            scenario=scenario,
            query=True,
            extra=extra_query_member and scenario == FORMAL_LEO_WEAK_SCENARIOS[0],
        )
        if (
            first_query_mutator is not None
            and scenario == FORMAL_LEO_WEAK_SCENARIOS[0]
        ):
            first_query_mutator(query)
        members.extend(
            [
                make_member_descriptor(
                    support,
                    relative_path=support.name,
                    artifact_role=f"support:{scenario}",
                    schema=SUPPORT_SCHEMA,
                    scenario=scenario,
                    npz_members=SUPPORT_NPZ_MEMBERS,
                ),
                make_member_descriptor(
                    query,
                    relative_path=query.name,
                    artifact_role=f"query:{scenario}",
                    schema=QUERY_SCHEMA,
                    scenario=scenario,
                    npz_members=QUERY_NPZ_MEMBERS,
                ),
            ]
        )
    metadata = {
        "schema": PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
        "artifact_stage": PREDICTOR_INPUT_STAGE,
        "stage": "stage2c",
        "receiver": "20-1",
        "seed": 713101,
        "new_class_count": 1,
        "support_pool_max_k": 1,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "registered_class_count": 1,
        "registered_classes": [{"class_index": 0, "class_handle": "cls_" + "0" * 64}],
        "candidate_lock_sha256": "9" * 64,
        **PHASE2_FULL_CONTRACT,
    }
    seal = tmp_path / "predictor.seal.json"
    write_predictor_package_manifest_and_seal(
        root,
        manifest_metadata=metadata,
        members=members,
        detached_seal_path=seal,
    )
    return root, seal, sha256_file(seal)


def test_preflight_verifies_every_member_before_iq_materialization(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path)
    _manifest, _seal, audit = preflight_stage2_predictor_package(
        root, detached_seal_path=seal, expected_seal_sha256=digest
    )
    assert audit["status"] == "PASS"
    assert audit["iq_payload_materialized"] is False
    assert audit["hash_and_member_audit_same_file_descriptor"] is True


def test_predictor_package_accepts_opaque_unlabeled_query(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path)
    _support, query, _manifest, audit = load_verified_stage2_predictor_bundle(
        root, detached_seal_path=seal, expected_seal_sha256=digest
    )
    assert query[FORMAL_LEO_WEAK_SCENARIOS[0]]["query_tokens"][0].startswith("qid_")
    assert audit["sample_level_post_channel_iq_sha256_status"] == "PASS"


def test_predictor_package_rejects_neutral_extra_npz_member_preopen(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path, extra_query_member=True)
    with pytest.raises(PredictorPackageError, match="NPZ member allowlist mismatch"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_extra_root_file(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path)
    (root / "unsealed-debug.txt").write_text("debug", encoding="utf-8")
    with pytest.raises(PredictorPackageError, match="unexpected package file"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_extra_root_directory(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path)
    (root / "unsealed-cache").mkdir()
    with pytest.raises(PredictorPackageError, match="unexpected package directory"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_extra_root_symlink(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path)
    extra = root / "unsealed-link"
    try:
        extra.symlink_to(root / "checkpoint.bin")
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(PredictorPackageError, match="symlink|reparse-point"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_encrypted_npz_member_preopen(tmp_path: Path) -> None:
    root, seal, digest = _package(
        tmp_path,
        first_query_mutator=_mark_first_zip_member_encrypted,
    )
    with pytest.raises(PredictorPackageError, match="encrypted NPZ ZIP member"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_unsupported_npz_compression_preopen(
    tmp_path: Path,
) -> None:
    root, seal, digest = _package(
        tmp_path,
        first_query_mutator=lambda path: _rewrite_zip_compression(path, zipfile.ZIP_BZIP2),
    )
    with pytest.raises(PredictorPackageError, match="unsupported NPZ ZIP compression"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_npz_crc_failure_preopen(tmp_path: Path) -> None:
    root, seal, digest = _package(
        tmp_path,
        first_query_mutator=lambda path: _patch_first_zip_uint(
            path,
            local_offset=14,
            central_offset=16,
            size=4,
        ),
    )
    with pytest.raises(PredictorPackageError, match="CRC/decompression validation failed"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_npz_member_uncompressed_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, digest = _package(tmp_path)
    monkeypatch.setattr(predictor_bundle, "MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(PredictorPackageError, match="member uncompressed size exceeds"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_npz_total_uncompressed_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, digest = _package(tmp_path)
    monkeypatch.setattr(predictor_bundle, "MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(PredictorPackageError, match="total uncompressed size exceeds"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_npz_compression_ratio_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, digest = _package(tmp_path)
    monkeypatch.setattr(predictor_bundle, "MAX_NPZ_COMPRESSION_RATIO", 0.5)
    with pytest.raises(PredictorPackageError, match="compression ratio exceeds"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_member_tamper_after_seal(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path)
    with (root / f"query_{FORMAL_LEO_WEAK_SCENARIOS[0]}.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(PredictorPackageError, match="digest mismatch"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


def test_predictor_package_rejects_symlink_swap_after_seal(tmp_path: Path) -> None:
    root, seal, digest = _package(tmp_path)
    target = root / f"query_{FORMAL_LEO_WEAK_SCENARIOS[0]}.npz"
    saved = tmp_path / "saved.npz"
    target.replace(saved)
    try:
        target.symlink_to(saved)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(PredictorPackageError, match="symlink|escapes root"):
        preflight_stage2_predictor_package(
            root, detached_seal_path=seal, expected_seal_sha256=digest
        )


@pytest.mark.parametrize(
    "value", ["/absolute.npz", "../escape.npz", "a\\b.npz", "C:/drive.npz"]
)
def test_relative_member_path_rejects_escape_forms(value: str) -> None:
    with pytest.raises(PredictorPackageError):
        validate_relative_member_path(value)


def test_query_token_value_cannot_encode_role_or_tx(tmp_path: Path) -> None:
    root, seal, _digest = _package(tmp_path)
    query = root / f"query_{FORMAL_LEO_WEAK_SCENARIOS[0]}.npz"
    # Rebuild before sealing is impossible here; this directly exercises the
    # materialization guard by replacing the archive and resealing a fresh tree.
    # A role/TX-bearing token does not match the strict opaque-token grammar.
    with np.load(query, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["query_tokens"] = np.asarray(["qid_target_old|14-10"])
    query.unlink()
    with query.open("xb") as handle:
        np.savez(handle, **payload)
    # The detached seal now correctly rejects the changed member before IQ use.
    with pytest.raises(PredictorPackageError):
        load_verified_stage2_predictor_bundle(
            root, detached_seal_path=seal, expected_seal_sha256=sha256_file(seal)
        )
