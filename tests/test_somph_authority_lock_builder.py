from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
TESTS_ROOT = REPO_ROOT / "tests"
for candidate in (str(CODE_ROOT), str(TESTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import test_somph_lineage_authority as authority_fixture  # noqa: E402
from cvsrffi import somph_lineage_authority as authority  # noqa: E402
from cvsrffi.somph_authority_lock_builder import (  # noqa: E402
    AUTHORITY_LOCK_BUILD_RECEIPT_NAME,
    SomphAuthorityLockBuildError,
    _locked_cache_spec_cell,
    _write_somph_authority_lock_package_impl,
)
from cvsrffi.somph_cache_build_matrix import write_cache_build_matrix  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
)
from cvsrffi.stage2_predictor_bundle import sha256_file  # noqa: E402
from training_controls import sat_channel_config_for_scenario  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _realistic_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    values = authority_fixture._fixture(tmp_path, monkeypatch)
    cache_set_path = Path(values["lock"]["cache_set_manifest"]["path"])
    cache_set = json.loads(cache_set_path.read_text(encoding="utf-8"))
    build_spec = json.loads(
        Path(values["build_spec_path"]).read_text(encoding="utf-8")
    )
    build_spec["role_specs"][0]["max_samples_per_tx"] = 3
    _write_json(Path(values["build_spec_path"]), build_spec)
    build_spec_sha = canonical_json_sha256(build_spec)
    cache_set["build_spec_sha256"] = build_spec_sha
    for scenario, audit in cache_set["cache_audits"].items():
        cache_path = cache_set_path.parent / cache_set["cache_npz_by_scenario"][
            scenario
        ]
        with np.load(cache_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        embedded = json.loads(str(arrays["manifest_json"].item()))
        channel_config = dict(sat_channel_config_for_scenario(scenario))
        channel_config.update(
            {
                "fs_hz": float(build_spec.get("sat_fs_hz", 25e6)),
                "fc_hz": float(build_spec.get("sat_fc_hz", 2.462e9)),
                "star_ground_channel_impl": "simplified_leo_residual",
            }
        )
        channel_sha = canonical_json_sha256(channel_config)
        overlays = [
            overlay_id(
                sample_id=str(sample_id),
                scenario=scenario,
                satellite_seed=int(seed),
                channel_config_sha256=channel_sha,
                iq_sha256=str(iq_sha),
            )
            for sample_id, seed, iq_sha in zip(
                arrays["sample_ids"],
                arrays["satellite_seeds"],
                arrays["post_channel_iq_sha256"],
            )
        ]
        arrays["overlay_ids"] = np.asarray(overlays)
        embedded["channel_config"] = channel_config
        embedded["channel_config_sha256"] = channel_sha
        embedded["build_spec_sha256"] = build_spec_sha
        embedded["overlay_ids_sha256"] = ids_sha256(overlays)
        embedded["role_inputs"][0]["physical_sample_count"] = (
            3 * len(values["lock"]["old_tx_ids"])
        )
        arrays["manifest_json"] = np.asarray(
            json.dumps(embedded, sort_keys=True)
        )
        cache_path.unlink()
        with cache_path.open("xb") as handle:
            np.savez(handle, **arrays)
        cache_sha = sha256_file(cache_path)
        cache_set["cache_sha256_by_scenario"][scenario] = cache_sha
        audit.update(
            {
                "path": str(cache_path),
                "sha256": cache_sha,
                "scenario": scenario,
                "row_count": len(values["lock"]["old_tx_ids"]),
                "physical_sample_ids_sha256": values["lock"][
                    "physical_sample_ids_sha256_by_scenario"
                ][scenario],
                "post_channel_iq_sha256_root": values["lock"][
                    "post_channel_iq_sha256_root_by_scenario"
                ][scenario],
                "overlay_ids_sha256": ids_sha256(overlays),
            }
        )
    _write_json(cache_set_path, cache_set)
    values["cache_set_path"] = cache_set_path
    values["exporter_path"] = Path(values["lock"]["exporter"]["path"])
    values["channel_members"] = {
        item["logical_name"]: Path(item["path"])
        for item in values["lock"]["channel_code_closure"]["members"]
    }
    cell_id = "rx_20_1_seed_713101"
    spec_manifest = {
        "schema": "cvs.phase2.somph_registered_cache_build_matrix.v2",
        "cache_scope": "stage2_target_old",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "formal_launch_authority": False,
        "required_samples_per_tx": 1,
        "support_pool_max_k": 1,
        "query_samples_per_tx": 0,
        "cells": [
            {
                "cell_id": cell_id,
                "receiver": "20-1",
                "seed": values["lock"]["seed"],
                "cache_scope": "stage2_target_old",
                "cache_output_root": str(cache_set_path.parent),
                "spec_path": Path(values["build_spec_path"]).name,
                "spec_file_sha256": sha256_file(values["build_spec_path"]),
                "spec_canonical_sha256": canonical_json_sha256(build_spec),
                "required_samples_per_tx": 1,
                "support_pool_max_k": 1,
                "query_samples_per_tx": 0,
            }
        ],
    }
    spec_manifest_path = tmp_path / "locked_cache_specs.json"
    _write_json(spec_manifest_path, spec_manifest)
    values["cache_spec_manifest_path"] = spec_manifest_path
    values["cache_spec_manifest_sha256"] = sha256_file(spec_manifest_path)
    values["cache_spec_cell_id"] = cell_id
    return values


def _refresh_locked_spec_manifest(values: dict) -> None:
    build_spec = json.loads(
        Path(values["build_spec_path"]).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        values["cache_spec_manifest_path"].read_text(encoding="utf-8")
    )
    manifest["cells"][0]["spec_file_sha256"] = sha256_file(
        values["build_spec_path"]
    )
    manifest["cells"][0]["spec_canonical_sha256"] = canonical_json_sha256(
        build_spec
    )
    _write_json(values["cache_spec_manifest_path"], manifest)
    values["cache_spec_manifest_sha256"] = sha256_file(
        values["cache_spec_manifest_path"]
    )


def _rewrite_cache(
    values: dict,
    scenario: str,
    mutate,
) -> None:
    cache_set = json.loads(
        values["cache_set_path"].read_text(encoding="utf-8")
    )
    cache_path = (
        values["cache_set_path"].parent
        / cache_set["cache_npz_by_scenario"][scenario]
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    embedded = json.loads(str(arrays["manifest_json"].item()))
    mutate(arrays, embedded)
    arrays["manifest_json"] = np.asarray(json.dumps(embedded, sort_keys=True))
    cache_path.unlink()
    with cache_path.open("xb") as handle:
        np.savez(handle, **arrays)
    cache_sha = sha256_file(cache_path)
    cache_set["cache_sha256_by_scenario"][scenario] = cache_sha
    cache_set["cache_audits"][scenario]["sha256"] = cache_sha
    for field in (
        "physical_sample_ids_sha256",
        "post_channel_iq_sha256_root",
        "overlay_ids_sha256",
    ):
        cache_set["cache_audits"][scenario][field] = embedded[field]
    _write_json(values["cache_set_path"], cache_set)


def _rewrite_build_spec_binding(values: dict, build_spec: dict) -> None:
    _write_json(Path(values["build_spec_path"]), build_spec)
    build_sha = canonical_json_sha256(build_spec)
    cache_set = json.loads(
        values["cache_set_path"].read_text(encoding="utf-8")
    )
    cache_set["build_spec_sha256"] = build_sha
    for scenario in cache_set["cache_npz_by_scenario"]:
        def mutate(_arrays, embedded, *, _sha=build_sha):
            embedded["build_spec_sha256"] = _sha

        _rewrite_cache(values, scenario, mutate)
        cache_set = json.loads(
            values["cache_set_path"].read_text(encoding="utf-8")
        )
    cache_set["build_spec_sha256"] = build_sha
    _write_json(values["cache_set_path"], cache_set)
    _refresh_locked_spec_manifest(values)


def _build(values: dict, output_root: Path) -> dict:
    return _write_somph_authority_lock_package_impl(
        values["cache_set_path"],
        cache_spec_manifest_path=values["cache_spec_manifest_path"],
        cache_spec_cell_id=values["cache_spec_cell_id"],
        exporter_path=values["exporter_path"],
        channel_code_members=values["channel_members"],
        output_root=output_root,
        expected_cache_spec_manifest_sha256=values[
            "cache_spec_manifest_sha256"
        ],
    )


def test_production_manifest_validation_accepts_actual_dynamic_sha_and_root(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "formal_matrix"
    cache_output_root = "/offline/formal/cache-output"
    manifest = write_cache_build_matrix(
        output_root=matrix_root,
        manysig_pkl="/datasets/ManySig.pkl",
        manytx_pkl="/datasets/ManyTx.pkl",
        cache_output_root=cache_output_root,
    )
    manifest_path = matrix_root / "manifest.json"

    cell, checked, spec_path, manifest_sha, manifest_size = (
        _locked_cache_spec_cell(
            manifest_path,
            cell_id="rx_20_1_seed_713101",
            require_exact_formal_manifest=True,
        )
    )

    assert checked == manifest
    assert cell["cache_output_root"] == (
        f"{cache_output_root}/rx_20_1/seed_713101"
    )
    assert spec_path == (matrix_root / cell["spec_path"]).absolute()
    assert manifest_sha == sha256_file(manifest_path)
    assert manifest_size == manifest_path.stat().st_size

    spec_path.write_text("{}", encoding="utf-8")
    with pytest.raises(
        SomphAuthorityLockBuildError,
        match="exact validation failed",
    ):
        _locked_cache_spec_cell(
            manifest_path,
            cell_id="rx_20_1_seed_713101",
            require_exact_formal_manifest=True,
        )


def test_builds_unsigned_real_byte_grounded_lock_and_bundle_accepts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    output_root = tmp_path / "unsigned_lock"
    result = _build(values, output_root)
    lock_path = output_root / authority.AUTHORITY_LOCK_NAME
    receipt_path = output_root / AUTHORITY_LOCK_BUILD_RECEIPT_NAME
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert result["authority_lock_sha256"] == sha256_file(lock_path)
    assert lock["receiver"] == "20-1"
    assert lock["cache_scope"] == "stage2_target_old"
    assert lock["old_tx_ids"] == values["lock"]["old_tx_ids"]
    assert lock["new_tx_ids"] == []
    assert lock["schema"] == "cvs.phase2.somph_leo_weak_authority_lock.v2"
    assert tuple(lock["physical_sample_ids_sha256_by_scenario"]) == (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )
    assert lock["physical_sample_scenario_assignment_sha256"]
    assert lock["cross_scenario_physical_disjointness_audit"] == "PASS"
    assert lock["single_observation_contract_audit"] == "PASS"
    for field, expected in authority.PHASE2_SINGLE_OBSERVATION_CONTRACT.items():
        assert lock[field] == expected
        assert receipt[field] == expected
    assert receipt["external_authority_lock_verified"] is False
    assert receipt["formal_launch_authority"] is False
    assert stat.S_IMODE(lock_path.stat().st_mode) & stat.S_IWUSR == 0
    assert stat.S_IMODE(receipt_path.stat().st_mode) & stat.S_IWUSR == 0

    envelope, public_key = authority_fixture._signed_envelope(
        lock,
        build_receipt_sha256=sha256_file(receipt_path),
        cache_spec_manifest_sha256=sha256_file(
            values["cache_spec_manifest_path"]
        ),
    )
    authority_fixture._install_test_envelope_verifier(monkeypatch, public_key)
    authority_fixture._install_test_build_authority_verifier(monkeypatch)
    envelope_path = tmp_path / "built_lock_envelope.json"
    _write_json(envelope_path, envelope)
    bundled = authority.write_somph_lineage_authority_bundle(
        lock_path,
        signed_authority_envelope_path=envelope_path,
        expected_signed_authority_envelope_sha256=sha256_file(envelope_path),
        authority_lock_build_receipt_path=receipt_path,
        cache_spec_manifest_path=values["cache_spec_manifest_path"],
        output_root=tmp_path / "built_lock_bundle",
    )
    assert bundled["external_authority_lock_verified"] is True
    assert bundled["formal_launch_authority"] is False
    authority.verify_somph_lineage_authority_bundle(
        bundled["authority_bundle_root"],
        expected_commit_sha256=bundled["authority_commit_sha256"],
    )


def test_rejects_build_spec_extra_query_truth_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    build_spec = json.loads(
        Path(values["build_spec_path"]).read_text(encoding="utf-8")
    )
    build_spec["query_labels"] = "forbidden"
    _write_json(Path(values["build_spec_path"]), build_spec)
    _refresh_locked_spec_manifest(values)
    with pytest.raises(
        SomphAuthorityLockBuildError, match="build spec exact schema drift"
    ):
        _build(values, tmp_path / "must_not_publish")
    assert not (tmp_path / "must_not_publish").exists()


def test_rejects_cache_bytes_that_drift_from_cache_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    cache_set = json.loads(values["cache_set_path"].read_text(encoding="utf-8"))
    scenario = next(iter(cache_set["cache_npz_by_scenario"]))
    cache_path = (
        values["cache_set_path"].parent
        / cache_set["cache_npz_by_scenario"][scenario]
    )
    cache_path.write_bytes(cache_path.read_bytes() + b"tamper")
    with pytest.raises(
        SomphAuthorityLockBuildError,
        match="self-declared cache SHA drift",
    ):
        _build(values, tmp_path / "must_not_publish")


def test_rejects_channel_member_order_or_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    reversed_members = dict(reversed(list(values["channel_members"].items())))
    with pytest.raises(
        SomphAuthorityLockBuildError, match="order/allowlist drift"
    ):
        _write_somph_authority_lock_package_impl(
            values["cache_set_path"],
            cache_spec_manifest_path=values["cache_spec_manifest_path"],
            cache_spec_cell_id=values["cache_spec_cell_id"],
            exporter_path=values["exporter_path"],
            channel_code_members=reversed_members,
            output_root=tmp_path / "must_not_publish",
            expected_cache_spec_manifest_sha256=values[
                "cache_spec_manifest_sha256"
            ],
        )


def test_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    output_root = tmp_path / "unsigned_lock"
    _build(values, output_root)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(values, output_root)


def test_rejects_self_consistent_satellite_seed_not_in_build_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    scenario = "leo_clear_weak"

    def mutate(arrays, embedded):
        new_seed = int(arrays["satellite_seeds"][0]) + 17
        arrays["satellite_seeds"] = np.full(
            arrays["satellite_seeds"].shape, new_seed, dtype=np.int64
        )
        embedded["role_satellite_seeds"] = {"target_old": new_seed}
        channel_sha = embedded["channel_config_sha256"]
        overlays = [
            overlay_id(
                sample_id=str(sample_id),
                scenario=scenario,
                satellite_seed=new_seed,
                channel_config_sha256=channel_sha,
                iq_sha256=str(iq_sha),
            )
            for sample_id, iq_sha in zip(
                arrays["sample_ids"], arrays["post_channel_iq_sha256"]
            )
        ]
        arrays["overlay_ids"] = np.asarray(overlays)
        embedded["overlay_ids_sha256"] = ids_sha256(overlays)

    _rewrite_cache(values, scenario, mutate)
    with pytest.raises(
        SomphAuthorityLockBuildError,
        match="satellite seed/build-spec drift",
    ):
        _build(values, tmp_path / "must_not_publish")


def test_rejects_self_consistent_noncanonical_channel_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    scenario = "leo_rain_weak"

    def mutate(arrays, embedded):
        config = dict(embedded["channel_config"])
        config["fs_hz"] = float(config["fs_hz"]) + 1.0
        channel_sha = canonical_json_sha256(config)
        embedded["channel_config"] = config
        embedded["channel_config_sha256"] = channel_sha
        overlays = [
            overlay_id(
                sample_id=str(sample_id),
                scenario=scenario,
                satellite_seed=int(seed),
                channel_config_sha256=channel_sha,
                iq_sha256=str(iq_sha),
            )
            for sample_id, seed, iq_sha in zip(
                arrays["sample_ids"],
                arrays["satellite_seeds"],
                arrays["post_channel_iq_sha256"],
            )
        ]
        arrays["overlay_ids"] = np.asarray(overlays)
        embedded["overlay_ids_sha256"] = ids_sha256(overlays)

    _rewrite_cache(values, scenario, mutate)
    with pytest.raises(
        SomphAuthorityLockBuildError,
        match="channel_config/fixed-code drift",
    ):
        _build(values, tmp_path / "must_not_publish")


def test_rejects_unbalanced_exact_coverage_and_role_input_count_lie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path / "coverage", monkeypatch)

    def mutate_count(arrays, _embedded):
        arrays["tx_ids"][0] = arrays["tx_ids"][1]

    _rewrite_cache(values, "leo_clear_weak", mutate_count)
    with pytest.raises(
        SomphAuthorityLockBuildError, match="exact per-role/TX/receiver"
    ):
        _build(values, tmp_path / "coverage_out")

    values = _realistic_fixture(tmp_path / "role_input", monkeypatch)
    for scenario in (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ):
        def mutate_role_input(_arrays, embedded):
            embedded["role_inputs"][0]["physical_sample_count"] += 1

        _rewrite_cache(values, scenario, mutate_role_input)
    with pytest.raises(
        SomphAuthorityLockBuildError,
        match="physical_sample_count drift",
    ):
        _build(values, tmp_path / "role_input_out")


def test_rejects_cross_scenario_reuse_even_with_recomputed_declared_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    source_scenario = "leo_clear_weak"
    target_scenario = "leo_rain_weak"
    cache_set = json.loads(
        values["cache_set_path"].read_text(encoding="utf-8")
    )
    source_path = (
        values["cache_set_path"].parent
        / cache_set["cache_npz_by_scenario"][source_scenario]
    )
    with np.load(source_path, allow_pickle=False) as archive:
        source_ids = np.asarray(archive["sample_ids"]).astype(str)
        source_dataset_sha = np.asarray(
            archive["source_dataset_sha256"]
        ).astype(str)
        source_record_indices = np.asarray(
            archive["source_record_indices"], dtype=np.int64
        )
        source_sig_ids = np.asarray(archive["sig_ids"]).astype(str)

    def reuse_physical_rows(arrays, embedded):
        arrays["sample_ids"] = source_ids.copy()
        arrays["source_dataset_sha256"] = source_dataset_sha.copy()
        arrays["source_record_indices"] = source_record_indices.copy()
        arrays["sig_ids"] = source_sig_ids.copy()
        embedded["physical_sample_ids_sha256"] = ids_sha256(
            source_ids.tolist()
        )
        overlays = [
            overlay_id(
                sample_id=str(sample_id),
                scenario=target_scenario,
                satellite_seed=int(seed),
                channel_config_sha256=embedded["channel_config_sha256"],
                iq_sha256=str(iq_sha),
            )
            for sample_id, seed, iq_sha in zip(
                arrays["sample_ids"],
                arrays["satellite_seeds"],
                arrays["post_channel_iq_sha256"],
            )
        ]
        arrays["overlay_ids"] = np.asarray(overlays)
        embedded["overlay_ids_sha256"] = ids_sha256(overlays)

    _rewrite_cache(values, target_scenario, reuse_physical_rows)
    cache_set = json.loads(
        values["cache_set_path"].read_text(encoding="utf-8")
    )
    ids_by_scenario = {}
    for scenario in cache_set["cache_npz_by_scenario"]:
        path = (
            values["cache_set_path"].parent
            / cache_set["cache_npz_by_scenario"][scenario]
        )
        with np.load(path, allow_pickle=False) as archive:
            ids_by_scenario[scenario] = (
                np.asarray(archive["sample_ids"]).astype(str).tolist()
            )
    cache_set["physical_sample_ids_sha256_by_scenario"] = {
        scenario: ids_sha256(ids_by_scenario[scenario])
        for scenario in ids_by_scenario
    }
    cache_set["physical_sample_scenario_assignment_sha256"] = (
        canonical_json_sha256(ids_by_scenario)
    )
    _write_json(values["cache_set_path"], cache_set)

    with pytest.raises(
        SomphAuthorityLockBuildError,
        match="physical samples overlap across LEO_weak scenarios",
    ):
        _build(values, tmp_path / "overlap_must_not_publish")


def test_relative_build_spec_paths_follow_spec_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    build_spec = json.loads(
        Path(values["build_spec_path"]).read_text(encoding="utf-8")
    )
    spec_dir = Path(values["build_spec_path"]).parent
    build_spec["role_specs"][0]["pkl"] = Path(
        build_spec["role_specs"][0]["pkl"]
    ).relative_to(spec_dir).as_posix()
    build_spec["out_manifest"] = Path(
        build_spec["out_manifest"]
    ).relative_to(spec_dir).as_posix()
    build_spec["out_npz_by_scenario"] = {
        scenario: Path(path).relative_to(spec_dir).as_posix()
        for scenario, path in build_spec["out_npz_by_scenario"].items()
    }
    _rewrite_build_spec_binding(values, build_spec)
    result = _build(values, tmp_path / "relative_lock")
    assert result["formal_launch_authority"] is False


def test_rejects_final_cache_set_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _realistic_fixture(tmp_path, monkeypatch)
    link = tmp_path / "cache_set_link.json"
    try:
        link.symlink_to(values["cache_set_path"])
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(SomphAuthorityLockBuildError):
        _write_somph_authority_lock_package_impl(
            link,
            cache_spec_manifest_path=values["cache_spec_manifest_path"],
            cache_spec_cell_id=values["cache_spec_cell_id"],
            exporter_path=values["exporter_path"],
            channel_code_members=values["channel_members"],
            output_root=tmp_path / "must_not_publish",
            expected_cache_spec_manifest_sha256=values[
                "cache_spec_manifest_sha256"
            ],
        )
