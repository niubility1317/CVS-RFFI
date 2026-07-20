from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA,
    LEO_WEAK_CACHE_SET_SCHEMA,
    LEO_WEAK_CACHE_STAGE,
    PHASE2_PHYSICAL_SAMPLE_OBSERVATION_POLICY,
    PHASE2_PHYSICAL_SAMPLE_ROOT_ID_POLICY,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
    physical_sample_id_from_values,
    post_channel_iq_sha256,
)
from cvsrffi import stage2_d96_d97_phase1_lodo as d97


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "export_phase1_singleobs_feature_archive.py"
)
SPEC = importlib.util.spec_from_file_location(
    "export_phase1_singleobs_feature_archive", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TinyRuntime(torch.nn.Module):
    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = rows.mean(dim=(1, 2)).unsqueeze(1)
        features = mean.repeat(1, 160)
        logits = torch.cat([mean, -mean], dim=1)
        return features, logits


def _runtime_lineage(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "adv3b02_runtime.pt"
    traced = torch.jit.trace(_TinyRuntime().eval(), torch.zeros((2, 2, 8)))
    torch.jit.save(traced, runtime)
    bundle_id = hashlib.sha256(b"phase1-bundle").hexdigest()
    dims = {
        "input_channels": 2,
        "z160": 160,
        "checkpoint_reference_logits": 2,
        "features": 288,
    }
    classes = ["tx0", "tx1"]
    receipt = {
        "schema": module.RUNTIME_EXPORT_RECEIPT_SCHEMA,
        "checkpoint_lineage_sha256": module.BASE_CHECKPOINT_SHA256,
        "runtime_sha256": _sha(runtime),
        "parity_status": "PASS",
        "max_abs_output_delta": 0.0,
        "parity_vector_root_sha256": hashlib.sha256(b"parity").hexdigest(),
        "runtime_archive_member_root_sha256": hashlib.sha256(b"archive").hexdigest(),
        "runtime_state_schema_root_sha256": hashlib.sha256(b"state-schema").hexdigest(),
        "runtime_state_bytes": runtime.stat().st_size,
        "runtime_structure_sha256": hashlib.sha256(b"structure").hexdigest(),
    }
    receipt_path = tmp_path / "runtime_export_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema": module.RUNTIME_MANIFEST_SCHEMA,
        "artifact_stage": "phase1_offline_before_target_access",
        "bundle_id": bundle_id,
        "phase1_checkpoint_sha256": module.BASE_CHECKPOINT_SHA256,
        "feature_runtime": {
            "path": runtime.name,
            "sha256": _sha(runtime),
            "schema": module.RUNTIME_SCHEMA,
        },
        "runtime_export_receipt": {
            "path": receipt_path.name,
            "sha256": _sha(receipt_path),
            "schema": module.RUNTIME_EXPORT_RECEIPT_SCHEMA,
        },
        "feature_dims": dims,
        "class_ids": classes,
    }
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    salt_receipt = {
        "schema": module.SELECTION_SALT_RECEIPT_SCHEMA,
        "status": "SEALED_BEFORE_TARGET_ACCESS",
        "artifact_stage": "phase1_offline_before_target_access",
        "bundle_id": bundle_id,
        "phase1_checkpoint_sha256": module.BASE_CHECKPOINT_SHA256,
        "selection_salt_sha256": hashlib.sha256(b"selection-salt").hexdigest(),
        "target_access": False,
    }
    salt_path = tmp_path / "selection_salt_receipt.json"
    salt_path.write_text(json.dumps(salt_receipt, sort_keys=True), encoding="utf-8")
    return runtime, manifest_path, salt_path, salt_receipt["selection_salt_sha256"]


def _identity_rows(scenario_index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    row_count = 4
    iq = (
        np.arange(row_count * 2 * 8, dtype=np.float32).reshape(row_count, 2, 8)
        + 100.0 * scenario_index
    ) / 100.0
    tx = np.asarray(["tx0", "tx1", "tx0", "tx1"])
    rx = np.asarray(["rx0", "rx0", "rx1", "rx1"])
    day = np.asarray(["d0", "d0", "d1", "d1"])
    eq = np.asarray(["1"] * row_count)
    sig = np.asarray([str(index) for index in range(row_count)])
    dataset_hash = hashlib.sha256(b"dataset").hexdigest()
    dataset_hashes = np.asarray([dataset_hash] * row_count)
    record_indices = np.arange(row_count, dtype=np.int64)
    sample_ids = np.asarray(
        [
            physical_sample_id_from_values(
                dataset_sha256=dataset_hash,
                source_record_index=index,
                role="source",
                tx_id=str(tx[index]),
                rx_id=str(rx[index]),
                day_id=str(day[index]),
                eq_id=str(eq[index]),
                sig_id=str(sig[index]),
            )
            for index in range(row_count)
        ]
    )
    return iq, {
        "tx_ids": tx,
        "rx_ids": rx,
        "day_ids": day,
        "eq_ids": eq,
        "sig_ids": sig,
        "source_dataset_sha256": dataset_hashes,
        "source_record_indices": record_indices,
        "sample_ids": sample_ids,
    }


def _write_verified_cache(path: Path, scenario: str, scenario_index: int) -> tuple[list[str], np.ndarray]:
    iq, identity = _identity_rows(scenario_index)
    sample_ids = identity["sample_ids"]
    row_count = len(sample_ids)
    channel_hash = canonical_json_sha256({"scenario": scenario, "test": True})
    iq_hashes = np.asarray([post_channel_iq_sha256(row) for row in iq])
    seeds = np.asarray([100 + scenario_index] * row_count, dtype=np.int64)
    overlay_ids = np.asarray(
        [
            overlay_id(
                sample_id=str(sample_ids[index]),
                scenario=scenario,
                satellite_seed=int(seeds[index]),
                channel_config_sha256=channel_hash,
                iq_sha256=str(iq_hashes[index]),
            )
            for index in range(row_count)
        ]
    )
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
        "phase2_physical_sample_observation_policy": PHASE2_PHYSICAL_SAMPLE_OBSERVATION_POLICY,
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_equalization_augmentation_transform_allowed": True,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_physical_sample_root_id_policy": PHASE2_PHYSICAL_SAMPLE_ROOT_ID_POLICY,
        "phase2_query_post_reception_view_fit_access": False,
        "builder_sha256": hashlib.sha256(b"builder").hexdigest(),
        "output_roles": ["source"],
        "sample_overlay_provenance_fields": [
            "sample_ids",
            "source_dataset_sha256",
            "source_record_indices",
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
        "channel_config_sha256": channel_hash,
        "physical_sample_ids_sha256": ids_sha256(sample_ids.astype(str).tolist()),
        "row_count": row_count,
    }
    payload = {
        "leo_weak_iq": iq,
        "raw_labels": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "domain_labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
        **identity,
        "dataset_role": np.asarray(["source"] * row_count),
        "channel_views": np.asarray(["rx_base"] * row_count),
        "sat_scenarios": np.asarray([scenario] * row_count),
        "satellite_seeds": seeds,
        "overlay_applied": np.asarray([True] * row_count),
        "post_channel_iq_sha256": iq_hashes,
        "overlay_ids": overlay_ids,
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    np.savez(path, **payload)
    return sample_ids.astype(str).tolist(), iq


def _real_cache_set(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    hashes: dict[str, str] = {}
    selected_sources: dict[str, np.ndarray] = {}
    shared_ids = None
    for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        path = tmp_path / f"{scenario}.npz"
        ids, iq = _write_verified_cache(path, scenario, index)
        shared_ids = ids if shared_ids is None else shared_ids
        assert ids == shared_ids
        mapping[scenario] = path.name
        hashes[scenario] = _sha(path)
        selected_sources[scenario] = iq
    manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "cache_scope": "source_validation",
        "output_roles": ["source"],
        "cache_npz_by_scenario": mapping,
        "cache_sha256_by_scenario": hashes,
        "physical_sample_ids_sha256": ids_sha256(shared_ids or []),
    }
    path = tmp_path / "cache_set.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, selected_sources


def _fake_cache(tmp_path: Path):
    cache_set, sources = _real_cache_set(tmp_path)
    arrays = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        _, identity = _identity_rows(scenario_index)
        arrays[scenario] = {
            "leo_weak_iq": sources[scenario],
            "sample_ids": identity["sample_ids"],
            "tx_ids": identity["tx_ids"],
            "rx_ids": identity["rx_ids"],
            "day_ids": identity["day_ids"],
            "dataset_role": np.asarray(["source"] * 4),
            "sat_scenarios": np.asarray([scenario] * 4),
        }
    payload = json.loads(cache_set.read_text(encoding="utf-8"))

    def loader(path, *, expected_scope, allowed_roles):
        assert expected_scope == "source_validation" and allowed_roles == {"source"}
        return arrays, payload, {"verified": True}

    return cache_set, arrays, loader


def _diagnostic_args(tmp_path: Path) -> dict[str, Any]:
    _, runtime_manifest, salt_receipt, _ = _runtime_lineage(tmp_path)
    cache_set, _, _ = _fake_cache(tmp_path)
    return {
        "cache_set_path": cache_set,
        "cache_set_sha256": _sha(cache_set),
        "runtime_manifest_path": runtime_manifest,
        "runtime_manifest_sha256": _sha(runtime_manifest),
        "selection_salt_receipt_path": salt_receipt,
        "selection_salt_receipt_sha256": _sha(salt_receipt),
        "output_dir": tmp_path / "out",
        "device": "cpu",
        "batch_size": 8,
    }


def _formal_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], str]:
    runtime_path, _runtime_manifest, salt_receipt, salt = _runtime_lineage(tmp_path)
    cache_set, _sources = _real_cache_set(tmp_path)
    runtime_sha = _sha(runtime_path)
    outer = hashlib.sha256(b"phase1-bundle").hexdigest()
    seal_sha = hashlib.sha256(b"formal-seal").hexdigest()
    envelope_sha = hashlib.sha256(b"formal-envelope").hexdigest()
    parity_sha = hashlib.sha256(b"formal-parity").hexdigest()
    expected = {
        "expected_detached_seal_sha256": seal_sha,
        "expected_signature_envelope_sha256": envelope_sha,
        "expected_checkpoint_lineage_sha256": module.BASE_CHECKPOINT_SHA256,
        "expected_runtime_sha256": runtime_sha,
        "expected_component_pre_sign_content_root_sha256": hashlib.sha256(
            b"component"
        ).hexdigest(),
        "expected_class_handle_binding_sha256": hashlib.sha256(b"classes").hexdigest(),
        "expected_parity_receipt_sha256": parity_sha,
        "expected_generation_lock_sha256": hashlib.sha256(b"generation").hexdigest(),
        "expected_method_lock_sha256": hashlib.sha256(b"method").hexdigest(),
        "expected_generation_config_sha256": hashlib.sha256(b"config").hexdigest(),
        "expected_generation_code_sha256": hashlib.sha256(b"code").hexdigest(),
        "expected_outer_content_root_sha256": outer,
    }
    loaded_runtime = torch.jit.load(str(runtime_path)).eval()
    context = {
        "formal_phase2_eligible": True,
        "outer_signature_verified": True,
        "detached_seal_verified": True,
        "runtime_checkpoint_parity_verified": True,
        "checkpoint_lineage_sha256": module.BASE_CHECKPOINT_SHA256,
        "runtime_sha256": runtime_sha,
        "outer_content_root_sha256": outer,
    }
    verified = module.deployment_bundle.VerifiedADV3B02DeploymentBundle(
        runtime=loaded_runtime,
        component=None,
        class_binding={
            "class_id_to_handle": [
                {"class_index": 0, "class_handle": "tx0"},
                {"class_index": 1, "class_handle": "tx1"},
            ]
        },
        parity_receipt={},
        generation_lock={},
        method_lock={},
        formal_phase2_context=context,
        audit={"status": "PASS"},
    )

    def verified_loader(package_root, **kwargs):
        assert Path(package_root) == tmp_path / "formal-bundle"
        assert kwargs["expected_outer_content_root_sha256"] == outer
        assert kwargs["expected_detached_seal_sha256"] == seal_sha
        return verified

    monkeypatch.setattr(
        module.deployment_bundle,
        "load_formal_adv3b02_deployment_bundle",
        verified_loader,
    )
    return {
        "cache_set_path": cache_set,
        "cache_set_sha256": _sha(cache_set),
        "package_root": tmp_path / "formal-bundle",
        "detached_seal_path": tmp_path / "formal.seal.json",
        "signature_envelope_path": tmp_path / "formal.signature.json",
        **expected,
        "selection_salt_receipt_path": salt_receipt,
        "selection_salt_receipt_sha256": _sha(salt_receipt),
        "output_dir": tmp_path / "formal",
        "device": "cpu",
        "batch_size": 8,
    }, salt


def _fake_forward(rows: np.ndarray):
    mean = rows.mean(axis=(1, 2), keepdims=False).astype(np.float32)[:, None]
    return np.repeat(mean, 160, axis=1), np.concatenate([mean, -mean], axis=1)


def test_formal_api_has_no_self_described_runtime_or_injection_and_diagnostic_is_never_formal(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(module.export_phase1_singleobs_feature_archive).parameters
    assert "forward_callback" not in parameters and "cache_loader" not in parameters
    assert "runtime_manifest_path" not in parameters
    assert "package_root" in parameters and "expected_outer_content_root_sha256" in parameters
    args = _diagnostic_args(tmp_path)
    _, _, loader = _fake_cache(tmp_path / "second-cache")
    result = module._export_test_diagnostic_not_formal(
        **args, forward_callback=_fake_forward, cache_loader=loader
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["formal_archive"] is False
    assert manifest["schema"] == module.DIAGNOSTIC_SCHEMA
    assert manifest["status"] == module.DIAGNOSTIC_STATUS

    with pytest.raises(
        module.Phase1SingleObservationArchiveError, match="known ADV3B02"
    ):
        module.export_development_phase1_singleobs_feature_archive(
            **args,
            expected_runtime_sha256=json.loads(
                Path(args["runtime_manifest_path"]).read_text(encoding="utf-8")
            )["feature_runtime"]["sha256"],
            expected_parity_receipt_sha256=json.loads(
                Path(args["runtime_manifest_path"]).read_text(encoding="utf-8")
            )["runtime_export_receipt"]["sha256"],
        )


def test_production_loader_torchscript_consumer_roundtrip_and_threeblock_golden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal_args, salt = _formal_args(tmp_path, monkeypatch)
    cache_set = Path(formal_args["cache_set_path"])
    _, sources = _real_cache_set(tmp_path)
    result = module.export_phase1_singleobs_feature_archive(**formal_args)
    with np.load(result["archive_path"], allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    assert tuple(arrays) == module.OUTPUT_MEMBER_ALLOWLIST
    assert arrays["features"].dtype == np.float32
    selected = []
    for row, physical_id in enumerate(arrays["physical_ids"].astype(str)):
        scenario = FORMAL_LEO_WEAK_SCENARIOS[module.selection_index(salt, physical_id)]
        selected.append(sources[scenario][row])
    selected_iq = np.stack(selected).astype(np.float32)
    z160, _ = _fake_forward(selected_iq)
    fft96 = module.spectral_logmag_sketch(selected_iq)
    rf32 = module.rf_statistics(selected_iq)
    golden = np.concatenate([z160, fft96, rf32], axis=1).astype(np.float32)
    assert np.allclose(arrays["features"], golden, atol=1e-6)
    normalized = d97.normalize_three_blocks(arrays["features"])
    manual = np.concatenate(
        [
            block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
            for block in (golden[:, :160], golden[:, 160:256], golden[:, 256:])
        ],
        axis=1,
    )
    manual /= np.linalg.norm(manual, axis=1, keepdims=True)
    assert np.allclose(normalized, manual, atol=1e-6)
    validated = d97.validate_feature_archive(result["archive_path"])
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert d97._validate_feature_archive_manifest(
        result["manifest_path"], result["manifest_sha256"], validated=validated
    )[
        "binding_kind"
    ] == "exporter_v2_file_array_and_runtime_lineage_sha256"
    assert manifest["lifecycle"]["phase2_bundle_ingest_allowed"] is False
    assert manifest["resolved_device"] == "cpu"


def test_development_sha_only_runtime_exports_nonformal_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manifest_path, legacy_salt_path, _salt = _runtime_lineage(
        tmp_path / "runtime"
    )
    cache_set, _sources = _real_cache_set(tmp_path / "cache")
    runtime_sha = _sha(runtime)
    monkeypatch.setattr(
        module,
        "KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256",
        frozenset({runtime_sha}),
    )
    monkeypatch.setattr(
        d97,
        "KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256",
        frozenset({runtime_sha}),
    )
    binding = module._load_known_runtime_sha_only(runtime, runtime_sha, ["tx0", "tx1"])
    salt_receipt = {
        "schema": module.SELECTION_SALT_RECEIPT_SCHEMA,
        "status": "SEALED_BEFORE_TARGET_ACCESS",
        "artifact_stage": "phase1_offline_before_target_access",
        "bundle_id": binding["bundle_id"],
        "phase1_checkpoint_sha256": module.BASE_CHECKPOINT_SHA256,
        "selection_salt_sha256": hashlib.sha256(b"sha-only-selection").hexdigest(),
        "target_access": False,
    }
    salt_path = tmp_path / "sha_only_salt.json"
    salt_path.write_text(json.dumps(salt_receipt, sort_keys=True), encoding="utf-8")
    result = module.export_development_sha_only_phase1_singleobs_feature_archive(
        cache_set_path=cache_set,
        cache_set_sha256=_sha(cache_set),
        runtime_path=runtime,
        expected_runtime_sha256=runtime_sha,
        class_ids=["tx0", "tx1"],
        selection_salt_receipt_path=salt_path,
        selection_salt_receipt_sha256=_sha(salt_path),
        output_dir=tmp_path / "sha_only_output",
        device="cpu",
        batch_size=8,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["formal_archive"] is False
    assert manifest["status"] == module.DEVELOPMENT_STATUS
    assert manifest["inputs"]["runtime_authority_mode"] == (
        module.DEVELOPMENT_SHA_ONLY_AUTHORITY_MODE
    )
    assert manifest["inputs"]["runtime_checkpoint_parity_receipt_sha256"] is None
    validated = d97.validate_feature_archive(result["archive_path"])
    receipt = d97._validate_feature_archive_manifest(
        result["manifest_path"], result["manifest_sha256"], validated=validated
    )
    assert receipt["development_lock_frozen"] is True
    assert receipt["full_phase1_lock"] is False

    runtime_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy = module.export_development_phase1_singleobs_feature_archive(
        cache_set_path=cache_set,
        cache_set_sha256=_sha(cache_set),
        runtime_manifest_path=manifest_path,
        runtime_manifest_sha256=_sha(manifest_path),
        expected_runtime_sha256=runtime_sha,
        expected_parity_receipt_sha256=runtime_manifest["runtime_export_receipt"][
            "sha256"
        ],
        selection_salt_receipt_path=legacy_salt_path,
        selection_salt_receipt_sha256=_sha(legacy_salt_path),
        output_dir=tmp_path / "legacy_development_output",
        device="cpu",
        batch_size=8,
    )
    assert legacy["formal_archive"] is False
    assert Path(legacy["archive_path"]).is_file()

    with pytest.raises(module.Phase1SingleObservationArchiveError, match="known SHA-bound"):
        module._load_known_runtime_sha_only(runtime, "0" * 64, ["tx0", "tx1"])


def test_selection_determinism_missing_scene_and_identity_drift(tmp_path: Path) -> None:
    args = _diagnostic_args(tmp_path)
    cache_set, arrays, loader = _fake_cache(tmp_path / "fake")
    args.update(cache_set_path=cache_set, cache_set_sha256=_sha(cache_set))
    first = module._export_test_diagnostic_not_formal(
        **args, forward_callback=_fake_forward, cache_loader=loader
    )
    with np.load(first["archive_path"], allow_pickle=False) as payload:
        scenarios = payload["scenario_names"].astype(str).tolist()
    assert scenarios == [
        FORMAL_LEO_WEAK_SCENARIOS[
            module.selection_index(
                json.loads(Path(args["selection_salt_receipt_path"]).read_text())["selection_salt_sha256"],
                physical_id,
            )
        ]
        for physical_id in arrays[FORMAL_LEO_WEAK_SCENARIOS[0]]["sample_ids"].astype(str)
    ]
    missing = dict(arrays)
    missing.pop(FORMAL_LEO_WEAK_SCENARIOS[-1])

    def missing_loader(*_args, **_kwargs):
        return missing, json.loads(cache_set.read_text()), {}

    with pytest.raises(module.Phase1SingleObservationArchiveError, match="all three"):
        module._export_test_diagnostic_not_formal(
            **{**args, "output_dir": tmp_path / "missing"},
            forward_callback=_fake_forward,
            cache_loader=missing_loader,
        )
    arrays[FORMAL_LEO_WEAK_SCENARIOS[1]]["rx_ids"][0] = "bad"
    with pytest.raises(module.Phase1SingleObservationArchiveError, match="TX/RX/day"):
        module._export_test_diagnostic_not_formal(
            **{**args, "output_dir": tmp_path / "drift"},
            forward_callback=_fake_forward,
            cache_loader=loader,
        )


def test_lineage_missing_cuda_unavailable_and_extra_member_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _diagnostic_args(tmp_path)
    _, _, loader = _fake_cache(tmp_path / "fake")
    runtime_manifest_path = Path(args["runtime_manifest_path"])
    broken = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    broken.pop("bundle_id")
    broken_path = tmp_path / "broken_runtime_manifest.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(module.Phase1SingleObservationArchiveError, match="exact schema"):
        module._export_test_diagnostic_not_formal(
            **{
                **args,
                "runtime_manifest_path": broken_path,
                "runtime_manifest_sha256": _sha(broken_path),
                "output_dir": tmp_path / "broken",
            },
            forward_callback=_fake_forward,
            cache_loader=loader,
        )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(module.Phase1SingleObservationArchiveError, match="CUDA requested"):
        module._export_test_diagnostic_not_formal(
            **{**args, "device": "cuda:0", "output_dir": tmp_path / "cuda"},
            forward_callback=_fake_forward,
            cache_loader=loader,
        )
    result = module._export_test_diagnostic_not_formal(
        **{**args, "output_dir": tmp_path / "valid"},
        forward_callback=_fake_forward,
        cache_loader=loader,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    with np.load(result["archive_path"], allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    tampered = tmp_path / "extra_member.npz"
    np.savez_compressed(tampered, **arrays, extra_debug_state=np.zeros(1))
    with pytest.raises(module.Phase1SingleObservationArchiveError, match="extra/missing"):
        module.verify_phase1_singleobs_archive(tampered, manifest)
