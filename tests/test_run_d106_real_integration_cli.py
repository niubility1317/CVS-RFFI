from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d106_real_integration.py"
)
RUNTIME_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "d106_candidate_runtime_manifest_20260801.json"
)
METHOD_LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "d106_rdce_method_lock_20260801.json"
)
SPEC = importlib.util.spec_from_file_location("run_d106_real_integration_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {
        "schema": runner.FIXTURE_SCHEMA,
        "candidate_id": runner.CANDIDATE_ID,
        "protocol_schema": runner.PROTOCOL_SCHEMA,
        "release_commit": "a" * 40,
        "method_lock_sha256": "b" * 64,
        "construction_code_sha256": "c" * 64,
    }
    for index, (path_name, hash_name) in enumerate(runner.PATH_HASH_FIELDS):
        path = (tmp_path / f"{path_name}.bin").resolve()
        payload = f"fixture-{index}".encode()
        path.write_bytes(payload)
        values[path_name] = str(path)
        values[hash_name] = _sha(payload)
    values["checkpoint_sha256"] = runner.EXPECTED_CHECKPOINT_SHA256
    checkpoint = Path(str(values["checkpoint"]))
    checkpoint.write_bytes(b"frozen-checkpoint")
    values["checkpoint_sha256"] = _sha(b"frozen-checkpoint")
    runtime_manifest = Path(str(values["runtime_manifest"]))
    runtime = {
        "schema": runner.RUNTIME_SCHEMA,
        "candidate_id": runner.CANDIDATE_ID,
        "protocol_schema": runner.PROTOCOL_SCHEMA,
        "method_lock_sha256": values["method_lock_sha256"],
        "phase1_tap_code_sha256": _sha(runner.PHASE1_TAP_CODE_PATH.read_bytes()),
        "construction_code_sha256": values["construction_code_sha256"],
        "integration_entry_code_sha256": _sha(
            runner.INTEGRATION_CODE_PATH.read_bytes()
        ),
        "model_augmentation_code_sha256": _sha(
            runner.MODEL_AUGMENTATION_PATH.read_bytes()
        ),
        "model_factory_code_sha256": _sha(runner.MODEL_FACTORY_PATH.read_bytes()),
        "model_backbone_code_sha256": _sha(
            runner.MODEL_BACKBONE_PATH.read_bytes()
        ),
        "checkpoint_sha256": values["checkpoint_sha256"],
        "source_split_manifest_sha256": values["source_split_manifest_sha256"],
        "upstream_source_pool_cache_set_sha256": values[
            "upstream_source_pool_cache_set_sha256"
        ],
        "storage_validator_schema": runner.LS_IQ_VALIDATOR_SCHEMA,
        "source_held_truth_access": False,
        "formal_query_access": False,
        "target_access": False,
        "performance_metrics_computed": False,
    }
    runtime_manifest.write_bytes(runner._canonical_bytes(runtime) + b"\n")
    values["runtime_sha256"] = _sha(runtime_manifest.read_bytes())
    fixture = (tmp_path / "fixture.json").resolve()
    fixture.write_bytes(runner._canonical_bytes(values))
    return fixture, values


def test_repository_runtime_manifest_binds_current_d106_implementation() -> None:
    lock_payload = METHOD_LOCK_PATH.read_bytes()
    method_lock = json.loads(lock_payload.decode("utf-8"))
    assert lock_payload == runner._canonical_bytes(method_lock) + b"\n"
    assert method_lock["candidate_id"] == runner.CANDIDATE_ID
    assert method_lock["protocol_schema"] == runner.PROTOCOL_SCHEMA
    assert method_lock["rank"] == 3
    assert method_lock["gamma"] == 0.2
    assert method_lock["k1_attenuation"] == [0.3, 0.3, 0.3]
    assert method_lock["query_fit"] is False
    assert method_lock["query_selection"] is False
    assert method_lock["query_update"] is False
    assert method_lock["formal_query_access"] is False
    assert method_lock["target_access"] is False

    payload = RUNTIME_MANIFEST_PATH.read_bytes()
    runtime = json.loads(payload.decode("utf-8"))
    assert set(runtime) == runner.RUNTIME_FIELDS
    assert payload == runner._canonical_bytes(runtime) + b"\n"
    assert runtime["candidate_id"] == runner.CANDIDATE_ID
    assert runtime["protocol_schema"] == runner.PROTOCOL_SCHEMA
    assert runtime["method_lock_sha256"] == _sha(
        runner._canonical_bytes(method_lock) + b"\n"
    )
    assert runtime["phase1_tap_code_sha256"] == _sha(
        runner.PHASE1_TAP_CODE_PATH.read_bytes()
    )
    assert runtime["construction_code_sha256"] == _sha(
        runner.CONSTRUCTION_CODE_PATH.read_bytes()
    )
    assert runtime["integration_entry_code_sha256"] == _sha(
        runner.INTEGRATION_CODE_PATH.read_bytes()
    )
    assert runtime["model_augmentation_code_sha256"] == _sha(
        runner.MODEL_AUGMENTATION_PATH.read_bytes()
    )
    assert runtime["model_factory_code_sha256"] == _sha(
        runner.MODEL_FACTORY_PATH.read_bytes()
    )
    assert runtime["model_backbone_code_sha256"] == _sha(
        runner.MODEL_BACKBONE_PATH.read_bytes()
    )
    assert runtime["source_held_truth_access"] is False
    assert runtime["formal_query_access"] is False
    assert runtime["target_access"] is False
    assert runtime["performance_metrics_computed"] is False


def test_fixture_rejects_extra_query_capability_and_actual_sha_drift(
    tmp_path: Path,
) -> None:
    fixture, values = _fixture(tmp_path)
    values["formal_query_path"] = str((tmp_path / "query").resolve())
    fixture.write_bytes(runner._canonical_bytes(values))
    with pytest.raises(runner.D106RealIntegrationError, match="semantic closure"):
        runner.load_fixture(fixture)

    fixture, values = _fixture(tmp_path / "second")
    Path(str(values["ls_archive"])).write_bytes(b"tamper")
    with pytest.raises(runner.D106RealIntegrationError, match="path/SHA256"):
        runner.load_fixture(fixture)


def test_fixture_rejects_nonimported_construction_code(
    tmp_path: Path,
) -> None:
    fixture, _values = _fixture(tmp_path)
    with pytest.raises(
        runner.D106RealIntegrationError, match="not the imported RDCE"
    ):
        runner.load_fixture(fixture)


def test_fixture_rejects_runtime_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, values = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "EXPECTED_CHECKPOINT_SHA256", values["checkpoint_sha256"]
    )
    monkeypatch.setattr(
        runner,
        "CONSTRUCTION_CODE_PATH",
        Path(str(values["construction_code"])).resolve(),
    )
    runtime_path = Path(str(values["runtime_manifest"]))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["formal_query_access"] = True
    runtime_path.write_bytes(runner._canonical_bytes(runtime) + b"\n")
    values["runtime_sha256"] = _sha(runtime_path.read_bytes())
    fixture.write_bytes(runner._canonical_bytes(values))
    with pytest.raises(runner.D106RealIntegrationError, match="runtime manifest"):
        runner.load_fixture(fixture)


@pytest.mark.parametrize(
    ("runtime_field", "error_pattern"),
    (
        ("model_augmentation_code_sha256", "D106 model augmentation code path/SHA256"),
        ("model_factory_code_sha256", "D106 model factory code path/SHA256"),
        ("model_backbone_code_sha256", "D106 model backbone code path/SHA256"),
    ),
)
def test_fixture_rejects_model_code_sha_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_field: str,
    error_pattern: str,
) -> None:
    fixture, values = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "EXPECTED_CHECKPOINT_SHA256", values["checkpoint_sha256"]
    )
    monkeypatch.setattr(
        runner,
        "CONSTRUCTION_CODE_PATH",
        Path(str(values["construction_code"])).resolve(),
    )
    runtime_path = Path(str(values["runtime_manifest"]))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[runtime_field] = "0" * 64
    runtime_path.write_bytes(runner._canonical_bytes(runtime) + b"\n")
    values["runtime_sha256"] = _sha(runtime_path.read_bytes())
    fixture.write_bytes(runner._canonical_bytes(values))
    with pytest.raises(runner.D106RealIntegrationError, match=error_pattern):
        runner.load_fixture(fixture)


@pytest.mark.parametrize(
    ("path_attribute", "error_pattern"),
    (
        ("MODEL_AUGMENTATION_PATH", "D106 model augmentation code"),
        ("MODEL_FACTORY_PATH", "D106 model factory code"),
        ("MODEL_BACKBONE_PATH", "D106 model backbone code"),
    ),
)
def test_fixture_rejects_missing_model_dependency_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_attribute: str,
    error_pattern: str,
) -> None:
    fixture, values = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "EXPECTED_CHECKPOINT_SHA256", values["checkpoint_sha256"]
    )
    monkeypatch.setattr(
        runner,
        "CONSTRUCTION_CODE_PATH",
        Path(str(values["construction_code"])).resolve(),
    )
    monkeypatch.setattr(
        runner,
        path_attribute,
        (tmp_path / f"missing-{path_attribute}.py").resolve(),
    )
    with pytest.raises(runner.D106RealIntegrationError, match=error_pattern):
        runner.load_fixture(fixture)


@pytest.mark.parametrize(
    "runtime_field",
    (
        "model_augmentation_code_sha256",
        "model_factory_code_sha256",
        "model_backbone_code_sha256",
    ),
)
def test_fixture_rejects_missing_model_runtime_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_field: str,
) -> None:
    fixture, values = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "EXPECTED_CHECKPOINT_SHA256", values["checkpoint_sha256"]
    )
    monkeypatch.setattr(
        runner,
        "CONSTRUCTION_CODE_PATH",
        Path(str(values["construction_code"])).resolve(),
    )
    runtime_path = Path(str(values["runtime_manifest"]))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.pop(runtime_field)
    runtime_path.write_bytes(runner._canonical_bytes(runtime) + b"\n")
    values["runtime_sha256"] = _sha(runtime_path.read_bytes())
    fixture.write_bytes(runner._canonical_bytes(values))
    with pytest.raises(runner.D106RealIntegrationError, match="expected SHA256 drift"):
        runner.load_fixture(fixture)


def test_real_integration_orchestrates_exact_no_query_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _values = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "EXPECTED_CHECKPOINT_SHA256", _values["checkpoint_sha256"]
    )
    monkeypatch.setattr(
        runner,
        "CONSTRUCTION_CODE_PATH",
        Path(str(_values["construction_code"])).resolve(),
    )
    calls: list[str] = []

    extracted = {
        "archive": str((tmp_path / "selected.npz").resolve()),
        "archive_sha256": "1" * 64,
        "receipt": str((tmp_path / "selected.json").resolve()),
        "receipt_sha256": "2" * 64,
        "validator_receipt": str((tmp_path / "validator.json").resolve()),
        "validator_receipt_sha256": "3" * 64,
    }
    tap = {
        "archive": str((tmp_path / "tap.npz").resolve()),
        "archive_sha256": "4" * 64,
        "receipt": str((tmp_path / "tap.json").resolve()),
        "receipt_sha256": "5" * 64,
    }
    asset = SimpleNamespace(
        lineage=SimpleNamespace(),
        asset_receipt_sha256="6" * 64,
        binding_sha256="7" * 64,
    )

    def fake_extract(**kwargs: object) -> dict[str, str]:
        calls.append("extract")
        assert "upstream_source_pool_cache_set" in kwargs
        Path(str(kwargs["output_dir"])).mkdir()
        return extracted

    def fake_export(**kwargs: object) -> dict[str, str]:
        calls.append("export")
        assert kwargs["selected_iq_archive_sha256"] == "1" * 64
        Path(str(kwargs["output_dir"])).mkdir()
        return tap

    def fake_tap_load(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append("load_tap")
        return SimpleNamespace(physical_ids=list(range(588)))

    def fake_build(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append("build_asset")
        assert type(kwargs["build_lock"]) is runner.D106RDCEBuildLock
        return asset

    def fake_save(value: object, directory: Path) -> dict[str, str]:
        calls.append("save_asset")
        directory.mkdir()
        (directory / "d106_rdce_asset.wire").write_bytes(b"wire")
        return {"wire_sha256": "8" * 64}

    def fake_load(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append("load_asset")
        return SimpleNamespace(
            asset_receipt_sha256=asset.asset_receipt_sha256,
            binding_sha256=asset.binding_sha256,
            basis_codes_qint8=np.zeros(
                (runner.RDCE_RANK, runner.Z_DIM), dtype=np.int8
            ),
        )

    monkeypatch.setattr(runner, "extract_d106_ls_received_iq", fake_extract)
    monkeypatch.setattr(runner, "export_d106_phase1_ls_tap", fake_export)
    monkeypatch.setattr(runner, "load_d106_phase1_ls_tap", fake_tap_load)
    monkeypatch.setattr(runner, "build_d106_rdce_asset", fake_build)
    monkeypatch.setattr(runner, "save_d106_rdce_asset", fake_save)
    monkeypatch.setattr(runner, "load_d106_rdce_asset", fake_load)

    output = tmp_path / "output"
    result = runner.run_real_integration(
        fixture_path=fixture,
        output_dir=output,
        device="cpu",
    )
    assert calls == [
        "extract",
        "export",
        "load_tap",
        "build_asset",
        "save_asset",
        "load_asset",
    ]
    assert result["status"] == "D106_REAL_INTEGRATION_COMPLETE_NO_QUERY"
    assert result["formal_query_access"] is False
    assert result["target_access"] is False
    assert result["performance_metrics_computed"] is False
    assert result["rdce_rank"] == runner.RDCE_RANK == 3
    assert set(path.name for path in output.iterdir()) == {
        "selected_ls_iq",
        "strict_tap",
        "rdce_asset",
        runner.RESULT_NAME,
        runner.COMPLETION_NAME,
    }
    result_bytes = (output / runner.RESULT_NAME).read_bytes()
    assert result_bytes == runner._canonical_bytes(json.loads(result_bytes))
    completion_bytes = (output / runner.COMPLETION_NAME).read_bytes()
    completion = json.loads(completion_bytes)
    assert completion_bytes == runner._canonical_bytes(completion)
    assert completion["result_sha256"] == _sha(result_bytes)
    assert completion["required_directories"] == [
        "selected_ls_iq",
        "strict_tap",
        "rdce_asset",
    ]
    assert completion["partial_output_acceptable"] is False
    with pytest.raises(FileExistsError):
        runner.run_real_integration(
            fixture_path=fixture,
            output_dir=output,
            device="cpu",
        )


def test_roundtrip_rank_gate_uses_wire_shape_not_dynamic_rank_attribute() -> None:
    exact = SimpleNamespace(
        basis_codes_qint8=np.zeros(
            (runner.RDCE_RANK, runner.Z_DIM), dtype=np.int8
        )
    )
    assert not hasattr(exact, "rank")
    assert runner._validated_roundtrip_rdce_rank(exact) == runner.RDCE_RANK

    with pytest.raises(
        runner.D106RealIntegrationError, match="roundtrip rank drift"
    ):
        runner._validated_roundtrip_rdce_rank(SimpleNamespace())
    with pytest.raises(
        runner.D106RealIntegrationError, match="roundtrip rank drift"
    ):
        runner._validated_roundtrip_rdce_rank(
            SimpleNamespace(
                basis_codes_qint8=np.zeros(
                    (runner.RDCE_RANK - 1, runner.Z_DIM), dtype=np.int8
                )
            )
        )


def test_parser_requires_fixture_and_output() -> None:
    parser = runner.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
