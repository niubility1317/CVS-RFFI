from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d106_real_integration.py"
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
    fixture = (tmp_path / "fixture.json").resolve()
    fixture.write_bytes(runner._canonical_bytes(values))
    return fixture, values


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
        rank=3,
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


def test_parser_requires_fixture_and_output() -> None:
    parser = runner.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
