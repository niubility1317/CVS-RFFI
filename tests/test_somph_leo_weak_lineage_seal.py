from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.somph_leo_weak_lineage_seal as lineage
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
from cvsrffi.somph_leo_weak_lineage_seal import (
    SomphLineageError,
    verify_somph_leo_weak_lineage_seal,
    write_somph_leo_weak_lineage_seal,
)
from cvsrffi.stage2_predictor_bundle import canonical_json_bytes, sha256_bytes


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    exporter = tmp_path / "exporter.py"
    exporter.write_text("print('export')\n", encoding="utf-8")
    channel = tmp_path / "channel.py"
    channel.write_text("def overlay(x): return x\n", encoding="utf-8")
    build_spec = {
        "schema": "cvs_leo_weak_iq_cache_build_spec_v1",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "satellite_seed_by_scenario": {
            scenario: 100 + index
            for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
        },
    }
    build_spec_path = tmp_path / "build_spec.json"
    _write_json(build_spec_path, build_spec)
    build_spec_sha = canonical_json_sha256(build_spec)
    exporter_sha = sha256_file(exporter)
    sample_ids = [
        physical_sample_id_from_values(
            role="target_old",
            tx_id=str(index),
            rx_id="20-1",
            day_id="1",
            eq_id="1",
            sig_id=str(index),
        )
        for index in range(2)
    ]
    physical_root = ids_sha256(sample_ids)
    cache_paths = {}
    cache_hashes = {}
    iq_roots = {}
    overlay_roots = {}
    channel_hashes = {}
    audits = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        iq = (
            np.arange(16, dtype=np.float32).reshape(2, 2, 4)
            + scenario_index
        )
        seeds = np.asarray([100 + scenario_index] * 2, dtype=np.int64)
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
                satellite_seed=int(seed),
                channel_config_sha256=channel_hash,
                iq_sha256=iq_hash,
            )
            for sample_id, seed, iq_hash in zip(sample_ids, seeds, iq_hashes)
        ]
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
            "build_spec_sha256": build_spec_sha,
            "output_roles": ["target_old"],
            "role_satellite_seeds": {"target_old": int(seeds[0])},
            "role_inputs": [{"role": "target_old", "dataset_sha256": "a" * 64}],
            "row_count": 2,
            "physical_sample_ids_sha256": physical_root,
            "post_channel_iq_sha256_root": ids_sha256(iq_hashes),
            "overlay_ids_sha256": ids_sha256(overlays),
            "channel_meta_keys": ["channel_model"],
            "sample_overlay_provenance_fields": [
                "sample_ids",
                "sat_scenarios",
                "satellite_seeds",
                "post_channel_iq_sha256",
                "overlay_ids",
            ],
        }
        cache = tmp_path / f"{scenario}.npz"
        with cache.open("xb") as handle:
            np.savez(
                handle,
                leo_weak_iq=iq,
                raw_labels=np.asarray([0, 1], dtype=np.int64),
                domain_labels=np.asarray([0, 0], dtype=np.int64),
                tx_ids=np.asarray(["0", "1"]),
                rx_ids=np.asarray(["20-1", "20-1"]),
                day_ids=np.asarray(["1", "1"]),
                eq_ids=np.asarray(["1", "1"]),
                sig_ids=np.asarray(["0", "1"]),
                dataset_role=np.asarray(["target_old", "target_old"]),
                channel_views=np.asarray(["rx_base", "rx_base"]),
                sat_scenarios=np.asarray([scenario, scenario]),
                satellite_seeds=seeds,
                overlay_applied=np.asarray([True, True], dtype=bool),
                sample_ids=np.asarray(sample_ids),
                post_channel_iq_sha256=np.asarray(iq_hashes),
                overlay_ids=np.asarray(overlays),
                manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
            )
        cache_paths[scenario] = cache.name
        cache_hashes[scenario] = sha256_file(cache)
        iq_roots[scenario] = ids_sha256(iq_hashes)
        overlay_roots[scenario] = ids_sha256(overlays)
        channel_hashes[scenario] = channel_hash
        audits[scenario] = {}
    cache_set = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_set_id": "test",
        "cache_scope": "stage2_target_old",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": ["target_old"],
        "cache_npz_by_scenario": cache_paths,
        "cache_sha256_by_scenario": cache_hashes,
        "cache_audits": audits,
        "physical_sample_ids_sha256": physical_root,
        "builder_sha256": exporter_sha,
        "build_spec_sha256": build_spec_sha,
        "build_spec_path_exposed_to_phase2": False,
    }
    cache_set_path = tmp_path / "cache_set.json"
    _write_json(cache_set_path, cache_set)
    channel_members = {"channel.py": channel}
    channel_closure = sha256_bytes(
        canonical_json_bytes(
            {
                "schema": lineage.CHANNEL_CODE_CLOSURE_SCHEMA,
                "members": [
                    {
                        "logical_name": "channel.py",
                        "sha256": sha256_file(channel),
                        "size_bytes": channel.stat().st_size,
                    }
                ],
            }
        )
    )
    return {
        "cache_set_manifest_path": cache_set_path,
        "expected_scope": "stage2_target_old",
        "expected_cache_set_manifest_sha256": sha256_file(cache_set_path),
        "expected_cache_sha256_by_scenario": cache_hashes,
        "exporter_path": exporter,
        "expected_exporter_sha256": exporter_sha,
        "build_spec_path": build_spec_path,
        "expected_build_spec_sha256": build_spec_sha,
        "channel_code_members": channel_members,
        "expected_channel_code_closure_sha256": channel_closure,
        "expected_channel_config_sha256_by_scenario": channel_hashes,
        "expected_physical_sample_ids_sha256": physical_root,
        "expected_post_channel_iq_sha256_root_by_scenario": iq_roots,
        "expected_overlay_ids_sha256_by_scenario": overlay_roots,
        "receipt_path": tmp_path / "receipt.json",
        "detached_seal_path": tmp_path / "receipt.seal.json",
    }


def test_writes_byte_grounded_receipt_and_detached_seal(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    receipt, seal = write_somph_leo_weak_lineage_seal(**kwargs)
    assert receipt["status"] == "BYTE_GROUNDED_SELF_CONSISTENCY_PASS"
    assert receipt["scenario_order"] == list(FORMAL_LEO_WEAK_SCENARIOS)
    assert receipt["manifest_hex_self_declaration_sufficient"] is False
    assert receipt["formal_launch_authority"] is False
    assert receipt["external_authority_lock_verified"] is False
    assert kwargs["receipt_path"].is_file()
    assert kwargs["detached_seal_path"].is_file()
    assert seal["receipt_sha256"] == sha256_file(kwargs["receipt_path"])
    receipt_text = kwargs["receipt_path"].read_text(encoding="utf-8")
    assert str(kwargs["build_spec_path"]) not in receipt_text
    assert "role_inputs" not in receipt_text
    verified, _verified_seal = verify_somph_leo_weak_lineage_seal(
        kwargs["receipt_path"],
        kwargs["detached_seal_path"],
        expected_detached_seal_sha256=sha256_file(kwargs["detached_seal_path"]),
    )
    assert verified == receipt


def test_receipt_without_matching_detached_seal_is_never_accepted(
    tmp_path: Path,
) -> None:
    kwargs = _fixture(tmp_path)
    write_somph_leo_weak_lineage_seal(**kwargs)
    with pytest.raises(SomphLineageError, match="seal SHA mismatch"):
        verify_somph_leo_weak_lineage_seal(
            kwargs["receipt_path"],
            kwargs["detached_seal_path"],
            expected_detached_seal_sha256="f" * 64,
        )


def test_channel_logical_name_cannot_smuggle_a_path(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    path = next(iter(kwargs["channel_code_members"].values()))
    kwargs["channel_code_members"] = {"../clean/dataset/channel.py": path}
    with pytest.raises(SomphLineageError, match="logical name is unsafe"):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_manifest_hex_cannot_replace_external_cache_hash(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    kwargs["expected_cache_sha256_by_scenario"] = {
        scenario: "f" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    with pytest.raises(SomphLineageError, match="declared SHA|external cache SHA"):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_exporter_and_build_spec_are_verified_from_real_bytes(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    kwargs["exporter_path"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SomphLineageError, match="external exporter SHA"):
        write_somph_leo_weak_lineage_seal(**kwargs)

    kwargs = _fixture(tmp_path / "build")
    kwargs["build_spec_path"].write_text('{"forged":"' + "a" * 64 + '"}', encoding="utf-8")
    with pytest.raises(SomphLineageError, match="canonical build-spec SHA"):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_channel_code_closure_is_verified_from_real_bytes(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    channel_path = next(iter(kwargs["channel_code_members"].values()))
    channel_path.write_text("def overlay(x): return 0\n", encoding="utf-8")
    with pytest.raises(SomphLineageError, match="channel-code closure SHA"):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_exact_scenario_order_is_required(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    values = kwargs["expected_cache_sha256_by_scenario"]
    kwargs["expected_cache_sha256_by_scenario"] = {
        FORMAL_LEO_WEAK_SCENARIOS[1]: values[FORMAL_LEO_WEAK_SCENARIOS[1]],
        FORMAL_LEO_WEAK_SCENARIOS[0]: values[FORMAL_LEO_WEAK_SCENARIOS[0]],
        FORMAL_LEO_WEAK_SCENARIOS[2]: values[FORMAL_LEO_WEAK_SCENARIOS[2]],
    }
    with pytest.raises(SomphLineageError, match="exact formal scenario order"):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_overlay_row_tamper_is_recomputed_not_self_declared(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    cache = tmp_path / f"{scenario}.npz"
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["overlay_ids"][0] = "e" * 64
    cache.unlink()
    with cache.open("xb") as handle:
        np.savez(handle, **payload)
    new_sha = sha256_file(cache)
    kwargs["expected_cache_sha256_by_scenario"][scenario] = new_sha
    cache_set = json.loads(
        kwargs["cache_set_manifest_path"].read_text(encoding="utf-8")
    )
    cache_set["cache_sha256_by_scenario"][scenario] = new_sha
    _write_json(kwargs["cache_set_manifest_path"], cache_set)
    kwargs["expected_cache_set_manifest_sha256"] = sha256_file(
        kwargs["cache_set_manifest_path"]
    )
    with pytest.raises(SomphLineageError, match="overlay row lineage"):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_cross_scenario_physical_order_drift_is_rejected(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    scenario = FORMAL_LEO_WEAK_SCENARIOS[-1]
    cache = tmp_path / f"{scenario}.npz"
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name in (
        "leo_weak_iq",
        "raw_labels",
        "domain_labels",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "eq_ids",
        "sig_ids",
        "dataset_role",
        "channel_views",
        "sat_scenarios",
        "satellite_seeds",
        "overlay_applied",
        "sample_ids",
        "post_channel_iq_sha256",
        "overlay_ids",
    ):
        payload[name] = payload[name][::-1]
    cache.unlink()
    with cache.open("xb") as handle:
        np.savez(handle, **payload)
    new_sha = sha256_file(cache)
    kwargs["expected_cache_sha256_by_scenario"][scenario] = new_sha
    cache_set = json.loads(
        kwargs["cache_set_manifest_path"].read_text(encoding="utf-8")
    )
    cache_set["cache_sha256_by_scenario"][scenario] = new_sha
    _write_json(kwargs["cache_set_manifest_path"], cache_set)
    kwargs["expected_cache_set_manifest_sha256"] = sha256_file(
        kwargs["cache_set_manifest_path"]
    )
    with pytest.raises(
        SomphLineageError, match="physical sample ordering|physical_sample_ids"
    ):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_npz_extra_member_and_compression_ratio_are_rejected(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    cache = tmp_path / f"{scenario}.npz"
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["clean_iq"] = np.zeros((2, 2, 4), dtype=np.float32)
    cache.unlink()
    with cache.open("xb") as handle:
        np.savez(handle, **payload)
    new_sha = sha256_file(cache)
    kwargs["expected_cache_sha256_by_scenario"][scenario] = new_sha
    cache_set = json.loads(
        kwargs["cache_set_manifest_path"].read_text(encoding="utf-8")
    )
    cache_set["cache_sha256_by_scenario"][scenario] = new_sha
    _write_json(kwargs["cache_set_manifest_path"], cache_set)
    kwargs["expected_cache_set_manifest_sha256"] = sha256_file(
        kwargs["cache_set_manifest_path"]
    )
    with pytest.raises(SomphLineageError, match="exact member allowlist"):
        write_somph_leo_weak_lineage_seal(**kwargs)

    kwargs = _fixture(tmp_path / "ratio")
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    cache = kwargs["cache_set_manifest_path"].parent / f"{scenario}.npz"
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["leo_weak_iq"] = np.zeros((2, 2, 1_000_000), dtype=np.float32)
    cache.unlink()
    with cache.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    new_sha = sha256_file(cache)
    kwargs["expected_cache_sha256_by_scenario"][scenario] = new_sha
    cache_set = json.loads(
        kwargs["cache_set_manifest_path"].read_text(encoding="utf-8")
    )
    cache_set["cache_sha256_by_scenario"][scenario] = new_sha
    _write_json(kwargs["cache_set_manifest_path"], cache_set)
    kwargs["expected_cache_set_manifest_sha256"] = sha256_file(
        kwargs["cache_set_manifest_path"]
    )
    with pytest.raises(SomphLineageError, match="compression ratio"):
        write_somph_leo_weak_lineage_seal(**kwargs)


def test_receipt_and_seal_are_no_overwrite(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    write_somph_leo_weak_lineage_seal(**kwargs)
    original_receipt = kwargs["receipt_path"].read_bytes()
    original_seal = kwargs["detached_seal_path"].read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_somph_leo_weak_lineage_seal(**kwargs)
    assert kwargs["receipt_path"].read_bytes() == original_receipt
    assert kwargs["detached_seal_path"].read_bytes() == original_seal
