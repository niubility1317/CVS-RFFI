from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import cvsrffi.stage2_d106_phase1_tap as d106


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(prefix: str, count: int) -> np.ndarray:
    return np.asarray([f"{prefix}{index:05d}" for index in range(count)], dtype=np.str_)


def _legal_ls_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the frozen 588-row 6x7x4/group14 source-label surface."""

    tx_labels: list[str] = []
    receiver_ids: list[str] = []
    day_ids: list[str] = []
    for receiver_index in range(7):
        for tx_index in range(6):
            for day_index, count in enumerate((4, 4, 3, 3)):
                tx_labels.extend([f"tx{tx_index}"] * count)
                receiver_ids.extend([f"rx{receiver_index}"] * count)
                day_ids.extend([f"day{day_index}"] * count)
    assert len(tx_labels) == 588
    return (
        np.asarray(tx_labels, dtype=np.str_),
        np.asarray(receiver_ids, dtype=np.str_),
        np.asarray(day_ids, dtype=np.str_),
        _text("p", 588),
    )


def _write_role_archives(
    root: Path, *, overlap_held: bool = False, object_labels: bool = False
) -> dict[str, Path]:
    ls_labels, ls_receivers, ls_days, ls_ids = _legal_ls_metadata()
    us_ids = _text("p", 5880)[588:]
    held_ids = _text("p", 8400)[5880:]
    if overlap_held:
        held_ids[0] = ls_ids[0]
    ls_dir = root / "L_s"
    us_dir = root / "U_s"
    held_dir = root / "scorer_only" / "source_val"
    ls_dir.mkdir(parents=True)
    us_dir.mkdir(parents=True)
    held_dir.mkdir(parents=True)
    role_payloads = {
        ls_dir / "features.npz": {
            "z_dom": np.zeros((588, 160), np.float32),
            "pre_relu": np.zeros((588, 160), np.float32),
            "receiver_ids": ls_receivers,
            "day_ids": ls_days,
            "tx_labels": np.asarray(
                ls_labels.tolist(), dtype=(object if object_labels else np.str_)
            ),
            "physical_ids": ls_ids,
        },
        us_dir / "features.npz": {
            "z_dom": np.zeros((5292, 160), np.float32),
            "receiver_ids": _text("r", 5292),
            "day_ids": _text("d", 5292),
            "physical_ids": us_ids,
        },
        held_dir / "features.npz": {
            "z_id": np.zeros((2520, 160), np.float32),
            "z_dom": np.zeros((2520, 160), np.float32),
            "pre_relu": np.zeros((2520, 160), np.float32),
            "labels": np.asarray(
                ["held-secret"] * 2520,
                dtype=(object if object_labels else np.str_),
            ),
            "receiver_ids": _text("r", 2520),
            "day_ids": _text("d", 2520),
            "physical_ids": held_ids,
            "scenario_names": np.asarray(["leo_clear_weak"] * 2520),
            "observation_ids": _text("o", 2520),
            "class_ids": np.asarray(["tx"]),
        },
    }
    for path, payload in role_payloads.items():
        if any(value.dtype.hasobject for value in payload.values()):
            np.savez(path, **payload)
        else:
            path.write_bytes(d106._deterministic_npz_bytes(payload))
    return {
        "L_s": ls_dir / "features.npz",
        "U_s": us_dir / "features.npz",
        "source_val": held_dir / "features.npz",
    }


def _write_split_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap_held: bool = False,
    object_labels: bool = False,
    cache_set_sha256: str = "2" * 64,
    salt_receipt_sha256: str = "3" * 64,
) -> tuple[Path, str, dict[str, Path]]:
    root = tmp_path / "split"
    archives = _write_role_archives(
        root, overlap_held=overlap_held, object_labels=object_labels
    )
    del monkeypatch
    with np.load(archives["L_s"], allow_pickle=False) as payload:
        ls_ids = payload["physical_ids"].astype(str).tolist()
    with np.load(archives["U_s"], allow_pickle=False) as payload:
        us_ids = payload["physical_ids"].astype(str).tolist()
    with np.load(archives["source_val"], allow_pickle=False) as payload:
        held_ids = payload["physical_ids"].astype(str).tolist()
    manifest = {
        "schema": d106.SOURCE_SPLIT_SCHEMA,
        "candidate_id": d106.D104_CANDIDATE_ID,
        "split_id": d106.SPLIT_ID,
        "status": d106.SOURCE_SPLIT_STATUS,
        "artifact_stage": "phase1_offline_before_new_source_held_truth_open",
        "protocol_schema": d106.PROTOCOL_SCHEMA,
        "target25_authorized": False,
        "target_access": False,
        "formal_query_access": False,
        "historical_exclusion_manifest": {
            "sha256": d106.EXCLUSION_MANIFEST_FILE_SHA256,
            "content_root_sha256": d106.EXCLUSION_MANIFEST_CONTENT_ROOT_SHA256,
            "query_count": d106.HISTORICAL_QUERY_COUNT,
        },
        "inputs": {
            "checkpoint_sha256": d106.EXPECTED_CHECKPOINT_SHA256,
            "runtime_sha256": "1" * 64,
            "source_train_cache_set_sha256": cache_set_sha256,
            "selection_salt_receipt_sha256": salt_receipt_sha256,
        },
        "partition": {
            "schema": "cvs.d104_r1.source_split.rows.v1",
            "candidate_id": d106.D104_CANDIDATE_ID,
            "split_id": d106.SPLIT_ID,
            "counts": dict(d106.EXPECTED_COUNTS),
            "physical_id_roots": {
                "L_s": d106._ordered_id_root(ls_ids),
                "U_s": d106._ordered_id_root(us_ids),
                "source_val": d106._ordered_id_root(held_ids),
            },
            "overlap_count": 0,
            "union_complete": True,
            "source_val_performance_computed": False,
        },
        "roles": {
            "L_s": {
                "archive": "L_s/features.npz",
                "archive_sha256": _sha(archives["L_s"]),
                "row_count": 588,
            },
            "U_s": {
                "archive": "U_s/features.npz",
                "archive_sha256": _sha(archives["U_s"]),
                "row_count": 5292,
            },
        },
        "source_val": {
            "scorer_archive": {
                "path": "scorer_only/source_val/features.npz",
                "sha256": _sha(archives["source_val"]),
            }
        },
    }
    manifest["partition"]["receipt_sha256"] = d106._d104_canonical_sha256(
        manifest["partition"]
    )
    path = root / "source_split_manifest.json"
    path.write_bytes(d106._canonical_bytes(manifest))
    return path, _sha(path), archives


def _build_disjoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_set_sha256: str = "2" * 64,
    salt_receipt_sha256: str = "3" * 64,
) -> tuple[Path, str, Path, str, dict[str, Path]]:
    manifest, manifest_sha, archives = _write_split_manifest(
        tmp_path,
        monkeypatch,
        cache_set_sha256=cache_set_sha256,
        salt_receipt_sha256=salt_receipt_sha256,
    )
    receipt = tmp_path / "disjoint.json"
    result = d106.build_d106_train_held_disjoint_receipt(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        output_path=receipt,
    )
    return manifest, manifest_sha, receipt, result["receipt_sha256"], archives


def _selected_source() -> tuple[dict[str, np.ndarray], np.ndarray]:
    _labels, receiver, day, physical = _legal_ls_metadata()
    metadata = {
        "receiver_ids": receiver,
        "day_ids": day,
        "physical_ids": physical,
        "scenario_names": np.asarray(["leo_clear_weak"] * 588),
        "observation_ids": _text("obs", 588),
    }
    return metadata, np.zeros((588, 2, 256), dtype=np.float32)


@pytest.mark.parametrize(
    ("raw", "parts"),
    [
        ("L_s\\features.npz", ("L_s", "features.npz")),
        (
            "scorer_only/source_val/features.npz",
            ("scorer_only", "source_val", "features.npz"),
        ),
    ],
)
def test_d104_manifest_relative_paths_are_portable(
    raw: str, parts: tuple[str, ...]
) -> None:
    assert d106._portable_manifest_relative_path(raw).parts == parts


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "/absolute/features.npz",
        "\\\\server\\share\\features.npz",
        "C:\\split\\features.npz",
        "../features.npz",
        "L_s/../features.npz",
        "L_s//features.npz",
        "L_s/./features.npz",
        "L_s:alias/features.npz",
    ],
)
def test_d104_manifest_relative_paths_reject_escape_and_ambiguity(raw: str) -> None:
    with pytest.raises(d106.D106Phase1TapError, match="path must be"):
        d106._portable_manifest_relative_path(raw)


def _write_synthetic_upstream_source_pool_cache_set(
    tmp_path: Path,
    *,
    root_name: str = "upstream-source-pool-cache",
    non_ls_variant: bool = False,
    invalid_non_ls: str | None = None,
) -> tuple[Path, str]:
    """Write a real 8400x3 cache with unreadable object label members."""

    cache_root = tmp_path / root_name
    cache_root.mkdir()
    physical_ids = _text("p", 8400)
    _labels, ls_receivers, ls_days, _ls_ids = _legal_ls_metadata()
    receiver_ids = np.concatenate(
        [ls_receivers, np.asarray(["rx-extra"] * (8400 - 588), dtype=np.str_)]
    )
    day_ids = np.concatenate(
        [ls_days, np.asarray(["day-extra"] * (8400 - 588), dtype=np.str_)]
    )
    scenario_map: dict[str, str] = {}
    hash_map: dict[str, str] = {}
    for scenario_index, scenario in enumerate(d106.FORMAL_LEO_WEAK_SCENARIOS):
        iq = np.full((8400, 2, 256), scenario_index + 0.25, dtype=np.float32)
        if non_ls_variant:
            iq[588:] += np.float32(17.0)
        iq_hashes = np.asarray(
            [d106.post_channel_iq_sha256(row) for row in iq], dtype=np.str_
        )
        seeds = np.asarray([100 + scenario_index] * 8400, dtype=np.int64)
        channel_hash = hashlib.sha256(f"channel:{scenario}".encode()).hexdigest()
        overlay_ids = np.asarray(
            [
                d106.overlay_id(
                    sample_id=str(physical_id),
                    scenario=scenario,
                    satellite_seed=100 + scenario_index,
                    channel_config_sha256=channel_hash,
                    iq_sha256=str(iq_hashes[row]),
                )
                for row, physical_id in enumerate(physical_ids.tolist())
            ],
            dtype=np.str_,
        )
        dataset_roles = np.asarray(["source"] * 8400, dtype=np.str_)
        if invalid_non_ls == "role":
            dataset_roles[588] = "invalid"
        elif invalid_non_ls == "hash":
            iq_hashes[588] = "f" * 64
        elif invalid_non_ls == "nan":
            iq[588, 0, 0] = np.float32(np.nan)
        elif invalid_non_ls is not None:
            raise AssertionError(f"unknown invalid_non_ls={invalid_non_ls}")
        inner = {
            "schema": d106.LEO_WEAK_CACHE_SCHEMA_V1,
            "artifact_stage": d106.LEO_WEAK_CACHE_STAGE,
            "phase2_sample_view_policy": d106.PHASE2_SAMPLE_VIEW_POLICY,
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
            "builder_sha256": hashlib.sha256(b"synthetic-builder").hexdigest(),
            "output_roles": ["source"],
            "sample_overlay_provenance_fields": list(
                d106.SOURCE_CACHE_PROVENANCE_BY_SCHEMA[
                    d106.LEO_WEAK_CACHE_SCHEMA_V1
                ]
            ),
            "channel_config_sha256": channel_hash,
            "physical_sample_ids_sha256": d106.ids_sha256(
                physical_ids.astype(str).tolist()
            ),
            "row_count": 8400,
        }
        payload = {
            "leo_weak_iq": iq,
            # Loading either member with allow_pickle=False is the sentinel.
            "raw_labels": np.asarray(["DO_NOT_ACCESS"] * 8400, dtype=object),
            "domain_labels": np.zeros(8400, dtype=np.int64),
            "tx_ids": np.asarray(["DO_NOT_ACCESS"] * 8400, dtype=object),
            "rx_ids": np.concatenate(
                [
                    receiver_ids[:588],
                    np.asarray(
                        ["rx-mutated" if non_ls_variant else "rx-extra"]
                        * (8400 - 588),
                        dtype=np.str_,
                    ),
                ]
            ),
            "day_ids": np.concatenate(
                [
                    day_ids[:588],
                    np.asarray(
                        ["day-mutated" if non_ls_variant else "day-extra"]
                        * (8400 - 588),
                        dtype=np.str_,
                    ),
                ]
            ),
            "eq_ids": np.asarray(["eq"] * 8400, dtype=np.str_),
            "sig_ids": _text("sig", 8400),
            "dataset_role": dataset_roles,
            "channel_views": np.asarray(["rx_base"] * 8400, dtype=np.str_),
            "sat_scenarios": np.asarray([scenario] * 8400, dtype=np.str_),
            "satellite_seeds": seeds,
            "overlay_applied": np.ones(8400, dtype=np.bool_),
            "sample_ids": physical_ids,
            "post_channel_iq_sha256": iq_hashes,
            "overlay_ids": overlay_ids,
            "manifest_json": np.asarray(
                d106._canonical_bytes(inner).decode("utf-8"), dtype=np.str_
            ),
        }
        assert tuple(payload) == d106.SOURCE_CACHE_REQUIRED_MEMBERS_V1
        cache_path = cache_root / f"{scenario}.npz"
        np.savez(cache_path, **payload)
        scenario_map[scenario] = cache_path.name
        hash_map[scenario] = _sha(cache_path)
    outer = {
        "schema": d106.LEO_WEAK_CACHE_SET_SCHEMA_V1,
        "artifact_stage": d106.LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": d106.PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "cache_scope": "source_validation",
        "output_roles": ["source"],
        "cache_npz_by_scenario": scenario_map,
        "cache_sha256_by_scenario": hash_map,
        "physical_sample_ids_sha256": d106.ids_sha256(
            physical_ids.astype(str).tolist()
        ),
    }
    cache_set = cache_root / "cache_set.json"
    cache_set.write_bytes(d106._canonical_bytes(outer))
    return cache_set, _sha(cache_set)


def _write_selection_salt_receipt(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "selection_salt_receipt.json"
    receipt = {
        "schema": "cvs.phase1.singleobs_selection_salt_receipt.v1",
        "status": "SEALED_BEFORE_TARGET_ACCESS",
        "artifact_stage": "phase1_offline_before_target_access",
        "bundle_id": "b" * 64,
        "phase1_checkpoint_sha256": d106.EXPECTED_CHECKPOINT_SHA256,
        "selection_salt_sha256": "4" * 64,
        "target_access": False,
    }
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return path, _sha(path)


def test_independent_disjoint_receipt_reads_ids_only_and_is_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, _archives = _write_split_manifest(
        tmp_path, monkeypatch, object_labels=True
    )
    receipt = tmp_path / "disjoint.json"
    result = d106.build_d106_train_held_disjoint_receipt(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        output_path=receipt,
    )
    receipt_sha = result["receipt_sha256"]
    binding = d106.load_d106_source_split_binding(manifest, manifest_sha)
    loaded = d106.load_d106_train_held_disjoint_receipt(
        receipt, receipt_sha, binding=binding
    )
    assert receipt.read_bytes() == d106._canonical_bytes(loaded)
    assert loaded["counts"] == {"L_s": 588, "U_s": 5292, "source_val": 2520}
    assert loaded["rho_label"] == 0.1
    assert loaded["phase1_train_count"] == 5880
    assert loaded["train_held_intersection_count"] == 0
    assert loaded["source_pool_union_count"] == 8400
    assert loaded["tx_labels_read"] is False
    assert "physical_ids" not in loaded
    assert "labels" not in loaded


def test_independent_disjoint_receipt_rejects_real_overlap_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, _archives = _write_split_manifest(
        tmp_path, monkeypatch, overlap_held=True
    )
    output = tmp_path / "never-written.json"
    with pytest.raises(d106.D106Phase1TapError, match="disjointness"):
        d106.build_d106_train_held_disjoint_receipt(
            source_split_manifest=manifest,
            source_split_manifest_sha256=manifest_sha,
            output_path=output,
        )
    assert not output.exists()


def test_source_split_rejects_count_sha_and_escape_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _manifest_sha, _archives = _write_split_manifest(tmp_path, monkeypatch)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["partition"]["counts"]["L_s"] = 587
    manifest.write_bytes(d106._canonical_bytes(value))
    with pytest.raises(d106.D106Phase1TapError, match="semantic closure"):
        d106.load_d106_source_split_binding(manifest, _sha(manifest))

    value["partition"]["counts"]["L_s"] = 588
    value["roles"]["L_s"]["archive"] = "../outside.npz"
    manifest.write_bytes(d106._canonical_bytes(value))
    with pytest.raises((d106.D106Phase1TapError, FileNotFoundError)):
        d106.load_d106_source_split_binding(manifest, _sha(manifest))

    with pytest.raises(d106.D106Phase1TapError, match="path/SHA256"):
        d106.load_d106_source_split_binding(manifest, "f" * 64)


def test_exact_ls_inner_join_preserves_ls_order_and_all_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest, _sha256, archives = _write_split_manifest(tmp_path, monkeypatch)
    metadata, iq = _selected_source()
    joined = d106.join_d106_ls_observations(
        metadata, iq, ls_archive=archives["L_s"]
    )
    assert joined.received_iq.shape == (588, 2, 256)
    assert joined.physical_ids.tolist() == _text("p", 588).tolist()
    assert joined.tx_labels.tolist() == _legal_ls_metadata()[0].tolist()
    assert joined.scenario_names.tolist() == ["leo_clear_weak"] * 588
    assert joined.observation_ids.tolist() == _text("obs", 588).tolist()
    assert not joined.received_iq.flags.writeable


@pytest.mark.parametrize("field", ["receiver_ids", "day_ids", "physical_ids"])
def test_exact_ls_inner_join_rejects_identity_or_tx_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    _manifest, _sha256, archives = _write_split_manifest(tmp_path, monkeypatch)
    metadata, iq = _selected_source()
    metadata[field][0] = "drift"
    with pytest.raises(d106.D106Phase1TapError, match=field):
        d106.join_d106_ls_observations(metadata, iq, ls_archive=archives["L_s"])


def test_fixed256_forward_zero_pads_three_same_iq_batches_and_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iq = np.arange(588 * 2 * 256, dtype=np.float32).reshape(588, 2, 256)
    seen: list[np.ndarray] = []

    def bridge(value, **_kwargs):
        seen.append(value.copy())
        return value

    class Tap:
        def __init__(self, value: np.ndarray) -> None:
            marker = value[:, 0, 0].reshape(-1, 1)
            self.pre_relu = np.repeat(marker, 160, axis=1).astype(np.float32)
            self.z_dom = (self.pre_relu + 1.0).astype(np.float32)

    monkeypatch.setattr(d106, "_tensor_from_d105_float32_c_iq", bridge)
    monkeypatch.setattr(d106, "extract_d105_feature_tap", lambda _model, value: Tap(value))
    model = type("EvalModel", (), {"training": False})()
    pre, dom, receipt = d106._forward_fixed256(model, iq, device="cpu")
    assert [batch.shape for batch in seen] == [(256, 2, 256)] * 3
    np.testing.assert_array_equal(seen[0], iq[:256])
    np.testing.assert_array_equal(seen[1], iq[256:512])
    np.testing.assert_array_equal(seen[2][:76], iq[512:])
    assert np.count_nonzero(seen[2][76:]) == 0
    assert pre.shape == (588, 160)
    np.testing.assert_array_equal(dom, pre + 1.0)
    assert receipt == {
        "forward_batch_capacity": 256,
        "forward_invocation_count": 3,
        "last_batch_real_rows": 76,
        "last_batch_padding_rows": 180,
        "same_iq_dual_forward": True,
        "fixed256_zero_pad_then_slice": True,
    }


def _extract_real_ls_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    non_ls_variant: bool = False,
    invalid_non_ls: str | None = None,
) -> tuple[dict[str, object], dict[str, Path]]:
    root.mkdir()
    cache_set, cache_sha = _write_synthetic_upstream_source_pool_cache_set(
        root, non_ls_variant=non_ls_variant, invalid_non_ls=invalid_non_ls
    )
    salt, salt_sha = _write_selection_salt_receipt(root)
    manifest, manifest_sha, disjoint, disjoint_sha, archives = _build_disjoint(
        root,
        monkeypatch,
        cache_set_sha256=cache_sha,
        salt_receipt_sha256=salt_sha,
    )
    result = d106.extract_d106_ls_received_iq(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        disjoint_receipt=disjoint,
        disjoint_receipt_sha256=disjoint_sha,
        upstream_source_pool_cache_set=cache_set,
        selection_salt_receipt=salt,
        output_dir=root / "selected-ls-iq",
    )
    return result, archives


def test_extract_isolates_588_artifact_from_legal_non_ls_iq_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _first_archives = _extract_real_ls_fixture(
        tmp_path / "first", monkeypatch
    )
    second, _second_archives = _extract_real_ls_fixture(
        tmp_path / "second", monkeypatch, non_ls_variant=True
    )
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert Path(str(first["archive"])).read_bytes() == Path(
        str(second["archive"])
    ).read_bytes()
    assert Path(str(first["receipt"])).read_bytes() == Path(
        str(second["receipt"])
    ).read_bytes()
    assert first["validator_receipt_sha256"] != second["validator_receipt_sha256"]
    validator = json.loads(
        Path(str(first["validator_receipt"])).read_text(encoding="utf-8")
    )
    assert validator["storage_iq_rows_read"] == 25200
    assert validator["storage_physical_rows_validated"] == 8400
    assert validator["selected_iq_rows_persisted"] == 588
    assert (
        validator["upstream_source_pool_cache_scope"]
        == d106.UPSTREAM_SOURCE_POOL_CACHE_SCOPE
    )
    assert len(validator["upstream_source_pool_cache_set_sha256"]) == 64
    assert (
        validator["d104_legacy_source_pool_hash_field"]
        == d106.D104_LEGACY_SOURCE_POOL_HASH_FIELD
    )
    assert validator["all_8400x3_storage_semantics_verified"] is True
    assert validator["validator_only_not_method_input"] is True
    assert tuple(validator["scenario_validation"]) == d106.FORMAL_LEO_WEAK_SCENARIOS
    assert {
        row["physical_sample_ids_sha256"]
        for row in validator["scenario_validation"].values()
    } == {next(iter(validator["scenario_validation"].values()))[
        "physical_sample_ids_sha256"
    ]}
    for row in validator["scenario_validation"].values():
        assert row["row_count"] == 8400
        assert row["full_storage_semantics_verified"] is True
        assert row["full_iq_digest_rows_verified"] == 8400
        assert row["full_overlay_rows_verified"] == 8400
    loaded_validator = d106.load_d106_ls_storage_validator(
        first["validator_receipt"],
        expected_sha256=str(first["validator_receipt_sha256"]),
        selected_archive_sha256=str(first["archive_sha256"]),
        selected_receipt_sha256=str(first["receipt_sha256"]),
        selected_content_root_sha256=validator["selected_content_root_sha256"],
    )
    assert loaded_validator["storage_validation_root_sha256"] == validator[
        "storage_validation_root_sha256"
    ]
    completion = json.loads(
        (Path(str(first["output_dir"])) / d106.COMPLETION_MARKER_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert completion["member_order"] == [
        d106.LS_IQ_ARCHIVE_NAME,
        d106.LS_IQ_RECEIPT_NAME,
        d106.LS_IQ_VALIDATOR_NAME,
    ]
    assert completion["partial_output_acceptable"] is False
    assert completion["directory_atomic_visibility_claimed"] is False
    with np.load(first["archive"], allow_pickle=False) as payload:
        assert tuple(payload.files) == d106.LS_IQ_MEMBERS
        assert len(payload["received_iq"]) == 588
    loaded = d106.load_d106_ls_received_iq(
        first["archive"],
        first["receipt"],
        expected_archive_sha256=str(first["archive_sha256"]),
        expected_receipt_sha256=str(first["receipt_sha256"]),
    )
    assert loaded.received_iq.shape == (588, 2, 256)
    assert loaded.receipt["contains_only_selected_ls_rows"] is True
    assert len(loaded.receipt["selected_content_root_sha256"]) == 64
    assert "upstream_source_pool_cache_set_sha256" not in loaded.receipt


def test_upstream_source_pool_rejects_legacy_name_as_actual_scope(
    tmp_path: Path,
) -> None:
    cache_set, _cache_sha = _write_synthetic_upstream_source_pool_cache_set(tmp_path)
    value = json.loads(cache_set.read_text(encoding="utf-8"))
    value["cache_scope"] = "source_train"
    cache_set.write_bytes(d106._canonical_bytes(value))
    with pytest.raises(d106.D106Phase1TapError, match="semantic closure"):
        d106._load_d106_source_cache_index(
            cache_set,
            expected_sha256=_sha(cache_set),
        )


@pytest.mark.parametrize("invalid_non_ls", ["role", "hash", "nan"])
def test_extract_rejects_invalid_non_ls_storage_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_non_ls: str,
) -> None:
    root = tmp_path / invalid_non_ls
    root.mkdir()
    cache_set, cache_sha = _write_synthetic_upstream_source_pool_cache_set(
        root, invalid_non_ls=invalid_non_ls
    )
    salt, salt_sha = _write_selection_salt_receipt(root)
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        root,
        monkeypatch,
        cache_set_sha256=cache_sha,
        salt_receipt_sha256=salt_sha,
    )
    output = root / "never-selected"
    with pytest.raises(d106.D106Phase1TapError):
        d106.extract_d106_ls_received_iq(
            source_split_manifest=manifest,
            source_split_manifest_sha256=manifest_sha,
            disjoint_receipt=disjoint,
            disjoint_receipt_sha256=disjoint_sha,
            upstream_source_pool_cache_set=cache_set,
            selection_salt_receipt=salt,
            output_dir=output,
        )
    assert not output.exists()


def test_storage_validator_rejects_wrong_sha_and_selected_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _archives = _extract_real_ls_fixture(tmp_path / "validator", monkeypatch)
    validator = json.loads(
        Path(str(result["validator_receipt"])).read_text(encoding="utf-8")
    )
    with pytest.raises(d106.D106Phase1TapError):
        d106.load_d106_ls_storage_validator(
            result["validator_receipt"],
            expected_sha256="0" * 64,
            selected_archive_sha256=str(result["archive_sha256"]),
            selected_receipt_sha256=str(result["receipt_sha256"]),
            selected_content_root_sha256=validator["selected_content_root_sha256"],
        )
    with pytest.raises(d106.D106Phase1TapError, match="semantic closure"):
        d106.load_d106_ls_storage_validator(
            result["validator_receipt"],
            expected_sha256=str(result["validator_receipt_sha256"]),
            selected_archive_sha256="0" * 64,
            selected_receipt_sha256=str(result["receipt_sha256"]),
            selected_content_root_sha256=validator["selected_content_root_sha256"],
        )


def test_formal_export_rejects_missing_or_wrong_storage_validator_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, archives = _extract_real_ls_fixture(tmp_path / "export-validator", monkeypatch)
    runtime = tmp_path / "runtime.json"
    runtime.write_bytes(b"{}")
    common = {
        "selected_iq_archive": result["archive"],
        "selected_iq_archive_sha256": str(result["archive_sha256"]),
        "selected_iq_receipt": result["receipt"],
        "selected_iq_receipt_sha256": str(result["receipt_sha256"]),
        "storage_validator_receipt": result["validator_receipt"],
        "ls_archive": archives["L_s"],
        "ls_archive_sha256": _sha(archives["L_s"]),
        "checkpoint": tmp_path / "missing-checkpoint.pth",
        "checkpoint_sha256": d106.EXPECTED_CHECKPOINT_SHA256,
        "runtime_manifest": runtime,
        "runtime_sha256": _sha(runtime),
        "output_dir": tmp_path / "never-tap",
    }
    with pytest.raises(d106.D106Phase1TapError):
        d106.export_d106_phase1_ls_tap(
            **common, storage_validator_receipt_sha256="0" * 64
        )
    Path(str(result["validator_receipt"])).unlink()
    with pytest.raises(d106.D106Phase1TapError):
        d106.export_d106_phase1_ls_tap(
            **common,
            storage_validator_receipt_sha256=str(
                result["validator_receipt_sha256"]
            ),
        )
    assert not (tmp_path / "never-tap").exists()


def test_completion_marker_is_required_and_partial_directories_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _archives = _extract_real_ls_fixture(tmp_path / "marker", monkeypatch)
    marker = Path(str(result["output_dir"])) / d106.COMPLETION_MARKER_NAME
    original_marker = marker.read_bytes()
    corrupted = json.loads(original_marker.decode("utf-8"))
    corrupted["member_sha256"][d106.LS_IQ_ARCHIVE_NAME] = "0" * 64
    marker.write_bytes(d106._canonical_bytes(corrupted))
    with pytest.raises(d106.D106Phase1TapError):
        d106.load_d106_ls_received_iq(
            result["archive"],
            result["receipt"],
            expected_archive_sha256=str(result["archive_sha256"]),
            expected_receipt_sha256=str(result["receipt_sha256"]),
        )
    marker.write_bytes(original_marker)
    marker.unlink()
    with pytest.raises(d106.D106Phase1TapError, match="completion marker"):
        d106.load_d106_ls_received_iq(
            result["archive"],
            result["receipt"],
            expected_archive_sha256=str(result["archive_sha256"]),
            expected_receipt_sha256=str(result["receipt_sha256"]),
        )
    partial = tmp_path / "partial-tap"
    partial.mkdir()
    (partial / d106.TAP_ARCHIVE_NAME).write_bytes(b"")
    (partial / d106.TAP_RECEIPT_NAME).write_bytes(b"")
    with pytest.raises(d106.D106Phase1TapError, match="completion marker"):
        d106.load_d106_phase1_ls_tap(
            partial / d106.TAP_ARCHIVE_NAME,
            partial / d106.TAP_RECEIPT_NAME,
            expected_archive_sha256=hashlib.sha256(b"").hexdigest(),
            expected_receipt_sha256=hashlib.sha256(b"").hexdigest(),
        )


@pytest.mark.parametrize(
    "callable_name",
    [
        "load_d106_ls_received_iq",
        "load_d106_ls_storage_validator",
        "_load_ls_join_metadata",
        "load_d105_exact_sha_bound_checkpoint",
        "build_d105_exact_model_from_checkpoint",
        "_forward_fixed256",
        "extract_d105_feature_tap",
        "_validate_checkpoint_loader_receipt",
        "_validate_model_reconstruction_receipt",
        "_validate_forward_receipt",
        "_deterministic_npz_bytes",
        "_write_new",
        "_write_completion_marker",
        "_load_completion_marker",
        "_publish_new_directory",
        "_read_regular_bytes",
    ],
)
def test_formal_export_rejects_critical_callable_replacement_before_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, callable_name: str
) -> None:
    monkeypatch.setattr(d106, callable_name, lambda *a, **k: None)
    output = tmp_path / "never"
    with pytest.raises(d106.D106Phase1TapError, match="execution closure drift"):
        d106.export_d106_phase1_ls_tap(
            selected_iq_archive=tmp_path / "missing.npz",
            selected_iq_archive_sha256="1" * 64,
            selected_iq_receipt=tmp_path / "missing.json",
            selected_iq_receipt_sha256="2" * 64,
            storage_validator_receipt=tmp_path / "missing-validator.json",
            storage_validator_receipt_sha256="5" * 64,
            ls_archive=tmp_path / "missing-ls.npz",
            ls_archive_sha256="3" * 64,
            checkpoint=tmp_path / "missing.pth",
            checkpoint_sha256=d106.EXPECTED_CHECKPOINT_SHA256,
            runtime_manifest=tmp_path / "missing-runtime.json",
            runtime_sha256="4" * 64,
            output_dir=output,
        )
    assert not output.exists()


def test_feature_export_signature_has_no_8400_cache_or_split_capability() -> None:
    parameters = set(inspect.signature(d106.export_d106_phase1_ls_tap).parameters)
    assert parameters == {
        "selected_iq_archive", "selected_iq_archive_sha256",
        "selected_iq_receipt", "selected_iq_receipt_sha256",
        "storage_validator_receipt", "storage_validator_receipt_sha256",
        "ls_archive", "ls_archive_sha256", "checkpoint",
        "checkpoint_sha256", "runtime_manifest", "runtime_sha256",
        "output_dir", "device",
    }
    assert parameters.isdisjoint(
        {
            "upstream_source_pool_cache_set", "source_split_manifest",
            "disjoint_receipt", "selection_salt_receipt",
        }
    )


@pytest.mark.parametrize(
    "callable_name",
    [
        "load_d105_tap_cache_selection_salt",
        "_load_ids_only",
        "select_d106_ls_cache_observations",
        "_load_d106_source_cache_index",
        "_load_selected_cache_scenario",
        "_load_inner_cache_manifest",
        "_npz_safe_member",
        "_resolve_cache_artifact",
        "_portable_manifest_relative_path",
        "_validate_ls_iq_arrays",
        "_deterministic_npz_bytes",
        "_write_new",
        "_write_completion_marker",
        "_load_completion_marker",
        "_publish_new_directory",
        "_read_regular_bytes",
    ],
)
def test_formal_extract_rejects_critical_callable_replacement_before_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, callable_name: str
) -> None:
    monkeypatch.setattr(d106, callable_name, lambda *a, **k: None)
    with pytest.raises(d106.D106Phase1TapError, match="execution closure drift"):
        d106.extract_d106_ls_received_iq(
            source_split_manifest=tmp_path / "missing-split.json",
            source_split_manifest_sha256="1" * 64,
            disjoint_receipt=tmp_path / "missing-disjoint.json",
            disjoint_receipt_sha256="2" * 64,
            upstream_source_pool_cache_set=tmp_path / "missing-cache.json",
            selection_salt_receipt=tmp_path / "missing-salt.json",
            output_dir=tmp_path / "never-extracted",
        )


def test_execution_subreceipts_reject_extra_keys_and_semantic_drift() -> None:
    checkpoint = {
        "policy": "weights_only_with_explicit_safe_globals",
        "torch_version": "test",
        "safe_globals_available": True,
        "weights_only": True,
        "exact_frozen_checkpoint_sha256_required": d106.EXPECTED_CHECKPOINT_SHA256,
        "caller_selected_checkpoint_allowed": False,
    }
    d106._validate_checkpoint_loader_receipt(
        checkpoint, checkpoint_sha256=d106.EXPECTED_CHECKPOINT_SHA256
    )
    with pytest.raises(d106.D106Phase1TapError, match="checkpoint-loader"):
        d106._validate_checkpoint_loader_receipt(
            checkpoint | {"extra": True},
            checkpoint_sha256=d106.EXPECTED_CHECKPOINT_SHA256,
        )
    model = {
        "loader": "d105_minimal_cvsincnet_checkpoint_reconstruction_v1",
        "model_factory": "model_dual_cvsincnet.build_dual_model",
        "backbone_factory": "model.build_model",
        "checkpoint_load_strict": True,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_mismatch": 0,
        "state_tensor_count": 195,
        "num_domains_from_state": 7,
        "input_len": 256,
        "eval_mode": True,
    }
    d106._validate_model_reconstruction_receipt(model)
    with pytest.raises(d106.D106Phase1TapError, match="model-reconstruction"):
        d106._validate_model_reconstruction_receipt(model | {"extra": 0})
    forward = {
        "forward_batch_capacity": 256,
        "forward_invocation_count": 3,
        "last_batch_real_rows": 76,
        "last_batch_padding_rows": 180,
        "same_iq_dual_forward": True,
        "fixed256_zero_pad_then_slice": True,
    }
    d106._validate_forward_receipt(forward)
    with pytest.raises(d106.D106Phase1TapError, match="forward receipt"):
        d106._validate_forward_receipt(forward | {"extra": 0})


def test_publish_rejects_posix_empty_directory_race(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "artifact").write_bytes(b"new")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        d106._publish_new_directory(staging, output, members=("artifact",))
    assert not any(output.iterdir())
    assert (staging / "artifact").read_bytes() == b"new"


def test_selected_iq_loader_rejects_archive_and_receipt_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _archives = _extract_real_ls_fixture(tmp_path / "symlink", monkeypatch)
    archive_link = tmp_path / "selected-link.npz"
    receipt_link = tmp_path / "selected-link.json"
    try:
        archive_link.symlink_to(Path(str(result["archive"])))
        receipt_link.symlink_to(Path(str(result["receipt"])))
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")
    with pytest.raises(d106.D106Phase1TapError, match="completed path|open failed"):
        d106.load_d106_ls_received_iq(
            archive_link,
            result["receipt"],
            expected_archive_sha256=str(result["archive_sha256"]),
            expected_receipt_sha256=str(result["receipt_sha256"]),
        )
    with pytest.raises(d106.D106Phase1TapError, match="completed path|open failed"):
        d106.load_d106_ls_received_iq(
            result["archive"],
            receipt_link,
            expected_archive_sha256=str(result["archive_sha256"]),
            expected_receipt_sha256=str(result["receipt_sha256"]),
        )


def test_real_checkpoint_cli_extract_export_validate_no_query(
    tmp_path: Path,
) -> None:
    """Opt-in real asset closure; never substitutes synthetic data for real."""

    fixture_path = os.environ.get("D106_REAL_INTEGRATION_FIXTURE")
    if not fixture_path:
        pytest.skip(
            "set D106_REAL_INTEGRATION_FIXTURE to a JSON contract containing "
            "manifest/disjoint/cache/salt/L_s/checkpoint paths and exact SHA256s"
        )
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    required = {
        "source_split_manifest", "source_split_manifest_sha256",
        "disjoint_receipt", "disjoint_receipt_sha256",
        "upstream_source_pool_cache_set",
        "selection_salt_receipt", "ls_archive", "ls_archive_sha256",
        "checkpoint", "checkpoint_sha256", "runtime_manifest", "runtime_sha256",
    }
    if type(fixture) is not dict or set(fixture) != required:
        pytest.fail("D106 real integration fixture key closure drift")
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / (
        "export_d106_phase1_ls_tap.py"
    )
    extracted_dir = tmp_path / "real-extracted"
    extracted = subprocess.run(
        [
            sys.executable, str(script), "extract",
            "--source-split-manifest", fixture["source_split_manifest"],
            "--source-split-manifest-sha256",
            fixture["source_split_manifest_sha256"],
            "--disjoint-receipt", fixture["disjoint_receipt"],
            "--disjoint-receipt-sha256", fixture["disjoint_receipt_sha256"],
            "--upstream-source-pool-cache-set",
            fixture["upstream_source_pool_cache_set"],
            "--selection-salt-receipt", fixture["selection_salt_receipt"],
            "--output-dir", str(extracted_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    extracted_result = json.loads(extracted.stdout)
    tap_dir = tmp_path / "real-tap"
    exported = subprocess.run(
        [
            sys.executable, str(script), "export",
            "--selected-iq-archive", extracted_result["archive"],
            "--selected-iq-archive-sha256", extracted_result["archive_sha256"],
            "--selected-iq-receipt", extracted_result["receipt"],
            "--selected-iq-receipt-sha256", extracted_result["receipt_sha256"],
            "--storage-validator-receipt", extracted_result["validator_receipt"],
            "--storage-validator-receipt-sha256",
            extracted_result["validator_receipt_sha256"],
            "--ls-archive", fixture["ls_archive"],
            "--ls-archive-sha256", fixture["ls_archive_sha256"],
            "--checkpoint", fixture["checkpoint"],
            "--checkpoint-sha256", fixture["checkpoint_sha256"],
            "--runtime-manifest", fixture["runtime_manifest"],
            "--runtime-sha256", fixture["runtime_sha256"],
            "--output-dir", str(tap_dir), "--device", "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    exported_result = json.loads(exported.stdout)
    validated = subprocess.run(
        [
            sys.executable, str(script), "validate",
            "--archive", exported_result["archive"],
            "--archive-sha256", exported_result["archive_sha256"],
            "--receipt", exported_result["receipt"],
            "--receipt-sha256", exported_result["receipt_sha256"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(validated.stdout)["row_count"] == 588


def _patch_export_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, np.ndarray],
    iq: np.ndarray,
    *,
    mock_selector: bool = True,
) -> None:
    del monkeypatch, metadata, iq, mock_selector
    pytest.skip("superseded: formal artifacts may not use patched critical callables")


def _superseded_export_and_loader_close_exact_members_and_derive_readonly_zid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch
    )
    metadata, iq = _selected_source()
    _patch_export_dependencies(monkeypatch, metadata, iq)
    output = tmp_path / "tap"
    result = d106.export_d106_phase1_ls_tap(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        disjoint_receipt=disjoint,
        disjoint_receipt_sha256=disjoint_sha,
        upstream_source_pool_cache_set=tmp_path / "cache.json",
        selection_salt_receipt=tmp_path / "salt.json",
        checkpoint=tmp_path / "checkpoint.pth",
        output_dir=output,
    )
    with np.load(result["archive"], allow_pickle=False) as payload:
        assert tuple(payload.files) == d106.TAP_MEMBERS
        assert not any("iq" in name.lower() for name in payload.files)
        assert "z_id" not in payload.files
    loaded = d106.load_d106_phase1_ls_tap(
        result["archive"],
        result["receipt"],
        expected_archive_sha256=result["archive_sha256"],
        expected_receipt_sha256=result["receipt_sha256"],
    )
    np.testing.assert_array_equal(loaded.z_id, np.maximum(loaded.pre_relu, 0.0))
    assert not loaded.z_id.flags.writeable
    assert not loaded.pre_relu.flags.writeable
    assert loaded.receipt["source_split_counts"] == d106.EXPECTED_COUNTS
    assert loaded.receipt["construction_closure"] == d106._construction_closure()
    assert len(
        loaded.receipt["construction_closure"][
            "construction_content_root_sha256"
        ]
    ) == 64
    assert set(
        loaded.receipt["construction_closure"]["files_sha256"]
    ) == {
        "d106_phase1_tap",
        "d106_selector_cache_parser",
        "d106_export_cli",
        "d105_feature_tap",
        "d105_exact_checkpoint_loader",
        "leo_weak_cache_primitives",
    }
    assert loaded.receipt["cache_storage_iq_rows_materialized"] == 25200
    assert loaded.receipt["cache_selection_domain_physical_rows"] == 588
    assert loaded.receipt["method_visible_received_iq_rows"] == 588
    assert loaded.receipt["method_visible_tx_label_rows"] == 588
    assert loaded.receipt["u_s_tx_labels_exposed"] is False
    assert loaded.receipt["source_val_tx_labels_exposed"] is False
    assert loaded.receipt["received_iq_persisted"] is False


def _superseded_export_uses_real_8400x3_cache_without_opening_source_label_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_set, cache_sha = _write_synthetic_upstream_source_pool_cache_set(tmp_path)
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch, cache_set_sha256=cache_sha
    )
    metadata, iq = _selected_source()
    _patch_export_dependencies(
        monkeypatch, metadata, iq, mock_selector=False
    )
    result = d106.export_d106_phase1_ls_tap(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        disjoint_receipt=disjoint,
        disjoint_receipt_sha256=disjoint_sha,
        upstream_source_pool_cache_set=cache_set,
        selection_salt_receipt=tmp_path / "salt.json",
        checkpoint=tmp_path / "checkpoint.pth",
        output_dir=tmp_path / "real-cache-tap",
    )
    loaded = d106.load_d106_phase1_ls_tap(
        result["archive"],
        result["receipt"],
        expected_archive_sha256=result["archive_sha256"],
        expected_receipt_sha256=result["receipt_sha256"],
    )
    assert loaded.tx_labels.tolist() == _legal_ls_metadata()[0].tolist()
    assert loaded.receipt["cache_storage_iq_rows_materialized"] == 25200
    assert loaded.receipt["cache_selection_domain_physical_rows"] == 588
    assert loaded.receipt["method_visible_received_iq_rows"] == 588
    assert loaded.receipt["method_visible_tx_label_rows"] == 588
    assert (
        loaded.receipt[
            "upstream_source_pool_tx_ids_or_raw_labels_read_or_materialized"
        ]
        is False
    )
    assert (
        loaded.receipt["u_s_tx_ids_or_raw_labels_read_or_materialized"] is False
    )
    assert (
        loaded.receipt["source_val_tx_ids_or_raw_labels_read_or_materialized"]
        is False
    )


def _superseded_export_join_failure_writes_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch
    )
    metadata, iq = _selected_source()
    metadata["receiver_ids"][0] = "wrong"
    _patch_export_dependencies(monkeypatch, metadata, iq)
    output = tmp_path / "tap-never-created"
    with pytest.raises(d106.D106Phase1TapError, match="receiver_ids"):
        d106.export_d106_phase1_ls_tap(
            source_split_manifest=manifest,
            source_split_manifest_sha256=manifest_sha,
            disjoint_receipt=disjoint,
            disjoint_receipt_sha256=disjoint_sha,
            upstream_source_pool_cache_set=tmp_path / "cache.json",
            selection_salt_receipt=tmp_path / "salt.json",
            checkpoint=tmp_path / "checkpoint.pth",
            output_dir=output,
        )
    assert not output.exists()


def _superseded_export_rejects_incomplete_source_pool_before_selection_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch
    )
    metadata, iq = _selected_source()
    _patch_export_dependencies(monkeypatch, metadata, iq)
    output = tmp_path / "tap-never-created"
    with pytest.raises(d106.D106Phase1TapError, match="8400x3"):
        d106.export_d106_phase1_ls_tap(
            source_split_manifest=manifest,
            source_split_manifest_sha256=manifest_sha,
            disjoint_receipt=disjoint,
            disjoint_receipt_sha256=disjoint_sha,
            upstream_source_pool_cache_set=tmp_path / "cache.json",
            selection_salt_receipt=tmp_path / "salt.json",
            checkpoint=tmp_path / "checkpoint.pth",
            output_dir=output,
        )
    assert not output.exists()


def _superseded_loader_rejects_noncanonical_receipt_and_archive_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch
    )
    metadata, iq = _selected_source()
    _patch_export_dependencies(monkeypatch, metadata, iq)
    result = d106.export_d106_phase1_ls_tap(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        disjoint_receipt=disjoint,
        disjoint_receipt_sha256=disjoint_sha,
        upstream_source_pool_cache_set=tmp_path / "cache.json",
        selection_salt_receipt=tmp_path / "salt.json",
        checkpoint=tmp_path / "checkpoint.pth",
        output_dir=tmp_path / "tap",
    )
    receipt = Path(result["receipt"])
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    with pytest.raises(d106.D106Phase1TapError, match="canonical"):
        d106.load_d106_phase1_ls_tap(
            result["archive"],
            receipt,
            expected_archive_sha256=result["archive_sha256"],
            expected_receipt_sha256=_sha(receipt),
        )
    archive = Path(result["archive"])
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(d106.D106Phase1TapError, match="path/SHA256"):
        d106.load_d106_phase1_ls_tap(
            archive,
            receipt,
            expected_archive_sha256=result["archive_sha256"],
            expected_receipt_sha256=_sha(receipt),
        )


def _superseded_loader_rejects_construction_closure_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch
    )
    metadata, iq = _selected_source()
    _patch_export_dependencies(monkeypatch, metadata, iq)
    result = d106.export_d106_phase1_ls_tap(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        disjoint_receipt=disjoint,
        disjoint_receipt_sha256=disjoint_sha,
        upstream_source_pool_cache_set=tmp_path / "cache.json",
        selection_salt_receipt=tmp_path / "salt.json",
        checkpoint=tmp_path / "checkpoint.pth",
        output_dir=tmp_path / "closure-tap",
    )
    receipt = Path(result["receipt"])
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["construction_closure"]["files_sha256"]["d105_feature_tap"] = "f" * 64
    receipt.write_bytes(d106._canonical_bytes(value))
    with pytest.raises(d106.D106Phase1TapError, match="semantic closure"):
        d106.load_d106_phase1_ls_tap(
            result["archive"],
            receipt,
            expected_archive_sha256=result["archive_sha256"],
            expected_receipt_sha256=_sha(receipt),
        )


def _superseded_export_and_loader_reject_output_and_artifact_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch
    )
    metadata, iq = _selected_source()
    _patch_export_dependencies(monkeypatch, metadata, iq)
    symlink_target = tmp_path / "existing-target"
    symlink_target.mkdir()
    output_link = tmp_path / "output-link"
    try:
        output_link.symlink_to(symlink_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    with pytest.raises(FileExistsError):
        d106.export_d106_phase1_ls_tap(
            source_split_manifest=manifest,
            source_split_manifest_sha256=manifest_sha,
            disjoint_receipt=disjoint,
            disjoint_receipt_sha256=disjoint_sha,
            upstream_source_pool_cache_set=tmp_path / "cache.json",
            selection_salt_receipt=tmp_path / "salt.json",
            checkpoint=tmp_path / "checkpoint.pth",
            output_dir=output_link,
        )
    result = d106.export_d106_phase1_ls_tap(
        source_split_manifest=manifest,
        source_split_manifest_sha256=manifest_sha,
        disjoint_receipt=disjoint,
        disjoint_receipt_sha256=disjoint_sha,
        upstream_source_pool_cache_set=tmp_path / "cache.json",
        selection_salt_receipt=tmp_path / "salt.json",
        checkpoint=tmp_path / "checkpoint.pth",
        output_dir=tmp_path / "symlink-loader-tap",
    )
    archive_link = tmp_path / "archive-link.npz"
    receipt_link = tmp_path / "receipt-link.json"
    try:
        archive_link.symlink_to(Path(result["archive"]))
        receipt_link.symlink_to(Path(result["receipt"]))
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")
    with pytest.raises(d106.D106Phase1TapError, match="archive path/SHA256"):
        d106.load_d106_phase1_ls_tap(
            archive_link,
            result["receipt"],
            expected_archive_sha256=result["archive_sha256"],
            expected_receipt_sha256=result["receipt_sha256"],
        )
    with pytest.raises(d106.D106Phase1TapError, match="receipt path/SHA256"):
        d106.load_d106_phase1_ls_tap(
            result["archive"],
            receipt_link,
            expected_archive_sha256=result["archive_sha256"],
            expected_receipt_sha256=result["receipt_sha256"],
        )


def _superseded_export_publish_race_preserves_newly_created_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha, disjoint, disjoint_sha, _archives = _build_disjoint(
        tmp_path, monkeypatch
    )
    metadata, iq = _selected_source()
    _patch_export_dependencies(monkeypatch, metadata, iq)
    output = tmp_path / "publish-race"
    original_rename = d106.os.rename

    def collide_then_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "keep.txt").write_text("keep", encoding="utf-8")
        original_rename(source, destination)

    monkeypatch.setattr(d106.os, "rename", collide_then_rename)
    with pytest.raises(OSError):
        d106.export_d106_phase1_ls_tap(
            source_split_manifest=manifest,
            source_split_manifest_sha256=manifest_sha,
            disjoint_receipt=disjoint,
            disjoint_receipt_sha256=disjoint_sha,
            upstream_source_pool_cache_set=tmp_path / "cache.json",
            selection_salt_receipt=tmp_path / "salt.json",
            checkpoint=tmp_path / "checkpoint.pth",
            output_dir=output,
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (output / d106.TAP_ARCHIVE_NAME).exists()
    assert not (output / d106.TAP_RECEIPT_NAME).exists()


def test_outputs_are_non_overwriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, manifest_sha, _archives = _write_split_manifest(tmp_path, monkeypatch)
    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        d106.build_d106_train_held_disjoint_receipt(
            source_split_manifest=manifest,
            source_split_manifest_sha256=manifest_sha,
            output_path=existing,
        )
    assert existing.read_text(encoding="utf-8") == "keep"
    existing_output = tmp_path / "existing-output"
    existing_output.mkdir()
    marker = existing_output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        d106.export_d106_phase1_ls_tap(
            selected_iq_archive=tmp_path / "unused-selected.npz",
            selected_iq_archive_sha256="1" * 64,
            selected_iq_receipt=tmp_path / "unused-selected.json",
            selected_iq_receipt_sha256="2" * 64,
            storage_validator_receipt=tmp_path / "unused-validator.json",
            storage_validator_receipt_sha256="5" * 64,
            ls_archive=tmp_path / "unused-ls.npz",
            ls_archive_sha256="3" * 64,
            checkpoint=tmp_path / "unused-checkpoint.pth",
            checkpoint_sha256=d106.EXPECTED_CHECKPOINT_SHA256,
            runtime_manifest=tmp_path / "unused-runtime.json",
            runtime_sha256="4" * 64,
            output_dir=existing_output,
        )
    assert marker.read_text(encoding="utf-8") == "keep"
