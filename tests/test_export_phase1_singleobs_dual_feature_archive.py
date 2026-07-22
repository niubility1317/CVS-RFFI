from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "export_phase1_singleobs_dual_feature_archive.py"
SPEC = importlib.util.spec_from_file_location("export_phase1_singleobs_dual_feature_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

OLD_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "test_export_phase1_singleobs_feature_archive.py"
OLD_SPEC = importlib.util.spec_from_file_location("old_singleobs_export_test_fixture", OLD_SCRIPT)
assert OLD_SPEC is not None and OLD_SPEC.loader is not None
old_fixture = importlib.util.module_from_spec(OLD_SPEC)
sys.modules[OLD_SPEC.name] = old_fixture
OLD_SPEC.loader.exec_module(old_fixture)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TinyDualRuntime(torch.nn.Module):
    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = rows.mean(dim=(1, 2)).unsqueeze(1)
        return mean.repeat(1, 160), (mean + 1.0).repeat(1, 160), torch.cat([mean, -mean], dim=1)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _lineage(tmp_path: Path, *, runtime_role: str = "candidate") -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "dual.ts"
    torch.jit.save(torch.jit.trace(_TinyDualRuntime().eval(), torch.zeros(2, 2, 8)), runtime)
    adapter_sha = hashlib.sha256(b"adapter").hexdigest()
    execution_contract = module._seal_graph_executor_optimize_false(torch.device("cpu"))
    export = _write_json(tmp_path / "export.json", {
        "schema": module.EXPORT_SCHEMA, "status": "PASS", "runtime_output_schema": module.RUNTIME_OUTPUT_SCHEMA,
        "checkpoint_sha256": module.BASE_CHECKPOINT_SHA256, "adapter_state_sha256": adapter_sha,
        "base_runtime_sha256": _sha(runtime) if runtime_role == "base" else hashlib.sha256(b"other").hexdigest(),
        "candidate_runtime_sha256": _sha(runtime) if runtime_role == "candidate" else hashlib.sha256(b"other").hexdigest(),
        "expected_input_len": 8, "runtime_batch_capacity": 256, "feature_dimensions": {"z_id": 160, "z_dom": 160, "tx_logits": 2},
        "formal_phase2_eligible": False, "bundle_created": False, "bundle_id": None,
        "execution_contract": execution_contract,
        "execution_contract_sha256": execution_contract["contract_sha256"],
        "max_abs_tolerance": 1.0e-5,
    })
    parity = _write_json(tmp_path / "parity.json", {
        "schema": module.PARITY_RECEIPT_SCHEMA, "status": "PASS", "runtime_output_schema": module.RUNTIME_OUTPUT_SCHEMA,
        "checkpoint_lineage_sha256": module.BASE_CHECKPOINT_SHA256, "adapter_state_sha256": adapter_sha,
        "runtime_sha256": _sha(runtime), "export_receipt_sha256": _sha(export), "runtime_role": runtime_role,
        "expected_input_len": 8, "expected_tx_classes": 2, "runtime_batch_capacity": 256,
        "runtime_invocations_per_parity_batch": 3, "max_abs_output_delta": 0.0,
        "runtime_calls_per_batch": 3,
        "execution_contract": execution_contract,
        "execution_contract_sha256": execution_contract["contract_sha256"],
        "max_abs_tolerance": 1.0e-5,
        "formal_phase2_eligible": False, "bundle_created": False, "bundle_id": None,
    })
    salt = _write_json(tmp_path / "salt.json", {
        "schema": module.SELECTION_SALT_RECEIPT_SCHEMA, "status": "SEALED_BEFORE_TARGET_ACCESS",
        "artifact_stage": "phase1_offline_before_target_access", "bundle_id": hashlib.sha256(b"phase1").hexdigest(),
        "phase1_checkpoint_sha256": module.BASE_CHECKPOINT_SHA256,
        "selection_salt_sha256": hashlib.sha256(b"salt").hexdigest(), "target_access": False,
    })
    return {"runtime": runtime, "export": export, "parity": parity, "salt": salt, "runtime_role": runtime_role}


def _fake_cache(tmp_path: Path) -> tuple[Path, dict[str, dict[str, np.ndarray]]]:
    npz_hashes = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        member = tmp_path / f"{scenario}.npz"
        member.write_bytes(f"verified-{scenario}".encode("utf-8"))
        npz_hashes[scenario] = _sha(member)
    cache_set = _write_json(tmp_path / "cache_set.json", {
        "cache_scope": "source_validation",
        "cache_npz_by_scenario": {scenario: f"{scenario}.npz" for scenario in FORMAL_LEO_WEAK_SCENARIOS},
        "cache_sha256_by_scenario": npz_hashes,
    })
    arrays: dict[str, dict[str, np.ndarray]] = {}
    physical = np.asarray(["p0", "p1", "p2", "p3"])
    tx = np.asarray(["tx0", "tx1", "tx0", "tx1"])
    rx = np.asarray(["rx0", "rx0", "rx1", "rx1"])
    day = np.asarray(["d0", "d0", "d1", "d1"])
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        rows = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8) + scenario_index * 100.0
        arrays[scenario] = {
            "leo_weak_iq": rows, "sample_ids": physical, "tx_ids": tx, "rx_ids": rx, "day_ids": day,
            "dataset_role": np.asarray(["source"] * 4), "sat_scenarios": np.asarray([scenario] * 4),
            "overlay_ids": np.asarray([f"overlay-{scenario}-{value}" for value in physical]),
        }
    return cache_set, arrays


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    lineage = _lineage(tmp_path)
    cache_set, arrays = _fake_cache(tmp_path)

    def loader(path, *, expected_scope, allowed_roles):
        assert Path(path) == cache_set and expected_scope == "source_validation" and allowed_roles == {"source"}
        return arrays, json.loads(cache_set.read_text(encoding="utf-8")), {
            "verified": True,
            "outer_observed_schema": module.LEO_WEAK_CACHE_SET_SCHEMA_V1,
            "inner_observed_schema_by_scenario": {
                scenario: module.LEO_WEAK_CACHE_SCHEMA_V1
                for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
            "legacy_schema_compatibility": True,
        }

    monkeypatch.setattr(module, "CACHE_LOADER", loader)
    monkeypatch.setattr(
        module, "KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256", frozenset({_sha(cache_set)})
    )
    return {
        "cache_set_path": cache_set, "cache_set_sha256": _sha(cache_set),
        "selection_salt_receipt_path": lineage["salt"], "selection_salt_receipt_sha256": _sha(lineage["salt"]),
        "runtime_path": lineage["runtime"], "runtime_sha256": _sha(lineage["runtime"]), "runtime_role": lineage["runtime_role"],
        "export_receipt_path": lineage["export"], "export_receipt_sha256": _sha(lineage["export"]),
        "parity_receipt_path": lineage["parity"], "parity_receipt_sha256": _sha(lineage["parity"]), "class_ids": ("tx0", "tx1"),
        "output_dir": tmp_path / "out", "device": "cpu", "batch_size": 2,
    }, arrays


def test_development_archive_preserves_explicit_registry_and_selected_overlay_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, arrays = _args(tmp_path, monkeypatch)
    result = module.export_phase1_singleobs_dual_feature_archive(**args)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    with np.load(result["archive_path"], allow_pickle=False) as archive:
        assert tuple(archive.files) == module.MEMBERS
        physical = archive["physical_ids"].astype(str).tolist()
        scenarios = archive["scenario_names"].astype(str).tolist()
        observation_ids = archive["observation_ids"].astype(str).tolist()
        assert archive["z_id"].shape == archive["z_dom"].shape == (4, 160)
        assert archive["tx_logits"].shape == (4, 2)
        assert archive["class_ids"].astype(str).tolist() == ["tx0", "tx1"]
    salt = json.loads(Path(args["selection_salt_receipt_path"]).read_text())["selection_salt_sha256"]
    expected_scenarios = [FORMAL_LEO_WEAK_SCENARIOS[module.selection_index(salt, value)] for value in physical]
    assert scenarios == expected_scenarios
    assert observation_ids == [f"overlay-{scenario}-{value}" for scenario, value in zip(scenarios, physical)]
    assert manifest["formal_phase2_eligible"] is False
    assert manifest["schema"] == module.SCHEMA
    assert manifest["access_audit"]["received_iq_persisted"] is False
    assert manifest["tx_logits_semantics"] == "raw_checkpoint_column_index_only_unbound_to_class_ids"
    assert manifest["held_runner_tx_logits_allowed"] is False
    assert tuple(manifest["inputs"]["cache_npz_sha256_by_scenario"]) == FORMAL_LEO_WEAK_SCENARIOS
    assert manifest["inputs"]["cache_outer_observed_schema"] == module.LEO_WEAK_CACHE_SET_SCHEMA_V1
    assert manifest["inputs"]["cache_legacy_schema_compatibility"] is True
    assert manifest["runtime_audit"] == {"same_iq_outputs": ["z_id", "z_dom", "tx_logits"], "single_runtime_call_per_selected_iq_batch": True, "runtime_invocations": 2, "batch_size": 2}
    assert manifest["inputs"]["execution_contract_sha256"] == manifest["inputs"]["execution_contract"]["contract_sha256"]
    assert set(manifest["array_sha256"]) == set(module.MEMBERS)
    module.verify_phase1_singleobs_dual_feature_archive(result["archive_path"], manifest)


def test_export_fails_closed_on_role_export_adapter_and_overlay_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, arrays = _args(tmp_path, monkeypatch)
    export = json.loads(Path(args["export_receipt_path"]).read_text())
    export["candidate_runtime_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    _write_json(Path(args["export_receipt_path"]), export)
    args["export_receipt_sha256"] = _sha(Path(args["export_receipt_path"]))
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="closure"):
        module.export_phase1_singleobs_dual_feature_archive(**args)

    args, arrays = _args(tmp_path / "adapter", monkeypatch)
    parity = json.loads(Path(args["parity_receipt_path"]).read_text())
    parity["adapter_state_sha256"] = hashlib.sha256(b"wrong-adapter").hexdigest()
    _write_json(Path(args["parity_receipt_path"]), parity)
    args["parity_receipt_sha256"] = _sha(Path(args["parity_receipt_path"]))
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="closure"):
        module.export_phase1_singleobs_dual_feature_archive(**args)

    args, arrays = _args(tmp_path / "overlay", monkeypatch)
    arrays[FORMAL_LEO_WEAK_SCENARIOS[0]].pop("overlay_ids")
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="overlay_ids"):
        module.export_phase1_singleobs_dual_feature_archive(**args)


def test_export_refuses_overwrite_and_rejects_extra_npz_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _arrays = _args(tmp_path, monkeypatch)
    result = module.export_phase1_singleobs_dual_feature_archive(**args)
    with pytest.raises(FileExistsError, match="overwrite"):
        module.export_phase1_singleobs_dual_feature_archive(**args)
    with np.load(result["archive_path"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    altered = tmp_path / "altered.npz"
    np.savez_compressed(altered, **arrays, forbidden=np.zeros(1, dtype=np.int8))
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="member"):
        module.verify_phase1_singleobs_dual_feature_archive(altered, manifest)


def test_explicit_class_registry_is_stable_across_label_row_order_and_cache_hashes_are_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, arrays = _args(tmp_path / "first", monkeypatch)
    first = module.export_phase1_singleobs_dual_feature_archive(**args)
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays[scenario]["tx_ids"] = np.asarray(["tx1", "tx0", "tx1", "tx0"])
    args["output_dir"] = tmp_path / "second" / "out"
    second = module.export_phase1_singleobs_dual_feature_archive(**args)
    with np.load(first["archive_path"], allow_pickle=False) as left, np.load(second["archive_path"], allow_pickle=False) as right:
        assert left["class_ids"].astype(str).tolist() == right["class_ids"].astype(str).tolist() == ["tx0", "tx1"]

    args, _ = _args(tmp_path / "hash-missing", monkeypatch)
    cache = json.loads(Path(args["cache_set_path"]).read_text())
    cache.pop("cache_sha256_by_scenario")
    _write_json(Path(args["cache_set_path"]), cache)
    args["cache_set_sha256"] = _sha(Path(args["cache_set_path"]))
    monkeypatch.setattr(
        module, "KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256", frozenset({args["cache_set_sha256"]})
    )
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="scenario hash"):
        module.export_phase1_singleobs_dual_feature_archive(**args)

    args, _ = _args(tmp_path / "hash-changed", monkeypatch)
    cache = json.loads(Path(args["cache_set_path"]).read_text())
    cache["cache_sha256_by_scenario"][FORMAL_LEO_WEAK_SCENARIOS[0]] = "0" * 64
    _write_json(Path(args["cache_set_path"]), cache)
    args["cache_set_sha256"] = _sha(Path(args["cache_set_path"]))
    monkeypatch.setattr(
        module, "KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256", frozenset({args["cache_set_sha256"]})
    )
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="NPZ hash/path"):
        module.export_phase1_singleobs_dual_feature_archive(**args)

    args, _ = _args(tmp_path / "outer-drift", monkeypatch)
    original_loader = module.CACHE_LOADER
    def drifting_loader(*loader_args, **loader_kwargs):
        value = original_loader(*loader_args, **loader_kwargs)
        Path(args["cache_set_path"]).write_text("{}", encoding="utf-8")
        return value
    monkeypatch.setattr(module, "CACHE_LOADER", drifting_loader)
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="changed during"):
        module.export_phase1_singleobs_dual_feature_archive(**args)


def test_runtime_call_count_atomic_cleanup_and_semantic_tamper_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _ = _args(tmp_path / "count", monkeypatch)
    calls = {"count": 0}
    class CountingRuntime:
        def eval(self): return self
        def __call__(self, rows):
            calls["count"] += 1
            mean = rows.mean(dim=(1, 2)).unsqueeze(1)
            return mean.repeat(1, 160), (mean + 1).repeat(1, 160), torch.cat([mean, -mean], dim=1)
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: CountingRuntime())
    result = module.export_phase1_singleobs_dual_feature_archive(**args)
    assert calls["count"] == 2

    with np.load(result["archive_path"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["physical_ids"] = arrays["physical_ids"].copy(); arrays["physical_ids"][1] = arrays["physical_ids"][0]
    semantic = tmp_path / "semantic.npz"
    np.savez_compressed(semantic, **arrays)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    manifest["array_sha256"] = {name: module._array_sha256(value) for name, value in arrays.items()}
    manifest["artifact"]["sha256"] = _sha(semantic)
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="physical"):
        module.verify_phase1_singleobs_dual_feature_archive(semantic, manifest)

    legacy = json.loads(json.dumps(manifest)); legacy["schema"] = "cvs.phase1.singleobs_dual_feature_archive.v1"
    with pytest.raises(module.Phase1SingleobsDualArchiveError, match="registry"):
        module.verify_phase1_singleobs_dual_feature_archive(result["archive_path"], legacy)

    args, _ = _args(tmp_path / "atomic", monkeypatch)
    monkeypatch.setattr(module, "verify_phase1_singleobs_dual_feature_archive", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("verify failed")))
    with pytest.raises(RuntimeError, match="verify failed"):
        module.export_phase1_singleobs_dual_feature_archive(**args)
    assert not Path(args["output_dir"]).exists()
    assert not list(Path(args["output_dir"]).parent.glob(".out.staging-*"))


def test_production_uses_real_v1_wrapper_and_manifest_hash_mapping_is_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_set, _sources = old_fixture._real_cache_set(
        tmp_path / "v1-cache",
        outer_schema=old_fixture.LEO_WEAK_CACHE_SET_SCHEMA_V1,
        inner_schema=old_fixture.LEO_WEAK_CACHE_SCHEMA_V1,
    )
    with pytest.raises(ValueError, match="schema"):
        module.load_verified_leo_weak_cache_set(
            cache_set, expected_scope="source_validation", allowed_roles={"source"}
        )
    lineage = _lineage(tmp_path / "lineage")
    monkeypatch.setattr(
        module,
        "KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256",
        frozenset({_sha(cache_set)}),
    )
    result = module.export_phase1_singleobs_dual_feature_archive(
        cache_set_path=cache_set,
        cache_set_sha256=_sha(cache_set),
        selection_salt_receipt_path=lineage["salt"],
        selection_salt_receipt_sha256=_sha(lineage["salt"]),
        runtime_path=lineage["runtime"],
        runtime_sha256=_sha(lineage["runtime"]),
        runtime_role=lineage["runtime_role"],
        export_receipt_path=lineage["export"],
        export_receipt_sha256=_sha(lineage["export"]),
        parity_receipt_path=lineage["parity"],
        parity_receipt_sha256=_sha(lineage["parity"]),
        class_ids=("tx0", "tx1"),
        output_dir=tmp_path / "v1-output",
        device="cpu",
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    inputs = manifest["inputs"]
    assert inputs["cache_outer_observed_schema"] == old_fixture.LEO_WEAK_CACHE_SET_SCHEMA_V1
    assert set(inputs["cache_inner_observed_schema_by_scenario"].values()) == {old_fixture.LEO_WEAK_CACHE_SCHEMA_V1}
    for mutation in (
        lambda value: value.pop(FORMAL_LEO_WEAK_SCENARIOS[0]),
        lambda value: {scenario: value[scenario] for scenario in reversed(FORMAL_LEO_WEAK_SCENARIOS)},
        lambda value: {**value, FORMAL_LEO_WEAK_SCENARIOS[0]: "not-a-sha"},
    ):
        altered = json.loads(json.dumps(manifest))
        hashes = altered["inputs"]["cache_npz_sha256_by_scenario"]
        changed = mutation(hashes)
        if changed is not None:
            altered["inputs"]["cache_npz_sha256_by_scenario"] = changed
        with pytest.raises(module.Phase1SingleobsDualArchiveError, match="legacy cache|SHA256"):
            module.verify_phase1_singleobs_dual_feature_archive(result["archive_path"], altered)
