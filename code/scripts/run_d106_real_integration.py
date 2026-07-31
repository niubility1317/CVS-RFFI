#!/usr/bin/env python3
"""Run the frozen D106 DATA-to-RDCE real-asset integration without queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d106_phase1_tap import (  # noqa: E402
    CANDIDATE_ID,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COUNTS,
    LS_IQ_VALIDATOR_SCHEMA,
    PROTOCOL_SCHEMA,
    extract_d106_ls_received_iq,
    export_d106_phase1_ls_tap,
    load_d106_phase1_ls_tap,
)
import cvsrffi.stage2_d106_phase1_tap as phase1_tap_module  # noqa: E402
from cvsrffi.stage2_d106_rdce_asset import (  # noqa: E402
    D106RDCEBuildLock,
    build_d106_rdce_asset,
    load_d106_rdce_asset,
    save_d106_rdce_asset,
)
import cvsrffi.stage2_d106_rdce_asset as rdce_asset_module  # noqa: E402


FIXTURE_SCHEMA = "cvs.d106.real_integration_fixture.v1"
RUNTIME_SCHEMA = "cvs.d106.runtime_manifest.v1"
RESULT_SCHEMA = "cvs.d106.real_integration_result.v1"
COMPLETION_SCHEMA = "cvs.d106.real_integration_completion.v1"
RESULT_NAME = "d106_real_integration_result.json"
COMPLETION_NAME = "COMPLETED.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PATH_HASH_FIELDS = (
    ("source_split_manifest", "source_split_manifest_sha256"),
    ("disjoint_receipt", "disjoint_receipt_sha256"),
    ("upstream_source_pool_cache_set", "upstream_source_pool_cache_set_sha256"),
    ("selection_salt_receipt", "selection_salt_receipt_sha256"),
    ("ls_archive", "ls_archive_sha256"),
    ("checkpoint", "checkpoint_sha256"),
    ("runtime_manifest", "runtime_sha256"),
    ("method_lock", "method_lock_sha256"),
    ("construction_code", "construction_code_sha256"),
)
CONSTRUCTION_CODE_PATH = Path(rdce_asset_module.__file__).resolve()
PHASE1_TAP_CODE_PATH = Path(phase1_tap_module.__file__).resolve()
INTEGRATION_CODE_PATH = Path(__file__).resolve()
MODEL_AUGMENTATION_PATH = (CODE_ROOT / "baseline_origin_sat_view.py").resolve()
MODEL_FACTORY_PATH = (CODE_ROOT / "model_dual_cvsincnet.py").resolve()
MODEL_BACKBONE_PATH = (CODE_ROOT / "model.py").resolve()
RUNTIME_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "method_lock_sha256",
    "phase1_tap_code_sha256",
    "construction_code_sha256",
    "integration_entry_code_sha256",
    "model_augmentation_code_sha256",
    "model_factory_code_sha256",
    "model_backbone_code_sha256",
    "checkpoint_sha256",
    "source_split_manifest_sha256",
    "upstream_source_pool_cache_set_sha256",
    "storage_validator_schema",
    "source_held_truth_access",
    "formal_query_access",
    "target_access",
    "performance_metrics_computed",
}
FIXTURE_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "release_commit",
    *(name for pair in PATH_HASH_FIELDS for name in pair),
}


class D106RealIntegrationError(RuntimeError):
    """Fail-closed error for the no-query integration runner."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, *, expected_sha256: str, name: str) -> bytes:
    if not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256):
        raise D106RealIntegrationError(f"{name} expected SHA256 drift")
    if path.is_symlink() or not path.is_absolute() or not path.is_file():
        raise D106RealIntegrationError(f"{name} must be an absolute regular file")
    before = path.stat()
    with path.open("rb") as handle:
        payload = handle.read()
        after = os.fstat(handle.fileno())
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or _sha256_bytes(payload) != expected_sha256
    ):
        raise D106RealIntegrationError(f"{name} path/SHA256 changed during read")
    return payload


def load_fixture(path: str | Path) -> tuple[Mapping[str, Any], str]:
    source = Path(path)
    if source.is_symlink() or not source.is_absolute() or not source.is_file():
        raise D106RealIntegrationError("fixture must be an absolute regular file")
    before = source.stat()
    with source.open("rb") as handle:
        payload = handle.read()
        after = os.fstat(handle.fileno())
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise D106RealIntegrationError("fixture changed during read")
    try:
        fixture = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RealIntegrationError("fixture must be strict UTF-8 JSON") from error
    if (
        type(fixture) is not dict
        or set(fixture) != FIXTURE_FIELDS
        or payload != _canonical_bytes(fixture)
        or fixture.get("schema") != FIXTURE_SCHEMA
        or fixture.get("candidate_id") != CANDIDATE_ID
        or fixture.get("protocol_schema") != PROTOCOL_SCHEMA
        or not isinstance(fixture.get("release_commit"), str)
        or not HEX40.fullmatch(fixture["release_commit"])
    ):
        raise D106RealIntegrationError("fixture semantic closure drift")
    for name in ("method_lock_sha256", "construction_code_sha256"):
        if not isinstance(fixture.get(name), str) or not HEX64.fullmatch(fixture[name]):
            raise D106RealIntegrationError(f"fixture {name} drift")
    bound_payloads: dict[str, bytes] = {}
    for path_name, hash_name in PATH_HASH_FIELDS:
        raw_path = fixture.get(path_name)
        expected = fixture.get(hash_name)
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise D106RealIntegrationError(f"fixture {path_name} binding drift")
        bound_payloads[path_name] = _read_regular(
            Path(raw_path), expected_sha256=expected, name=path_name
        )
    if Path(fixture["construction_code"]).resolve() != CONSTRUCTION_CODE_PATH:
        raise D106RealIntegrationError(
            "fixture construction code is not the imported RDCE implementation"
        )
    if fixture["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise D106RealIntegrationError("fixture checkpoint is not the frozen D106 model")
    try:
        runtime = json.loads(bound_payloads["runtime_manifest"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RealIntegrationError(
            "D106 runtime manifest must be strict UTF-8 JSON"
        ) from error
    expected_runtime = {
        "schema": RUNTIME_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "method_lock_sha256": fixture["method_lock_sha256"],
        "phase1_tap_code_sha256": _sha256_bytes(
            _read_regular(
                PHASE1_TAP_CODE_PATH,
                expected_sha256=runtime.get("phase1_tap_code_sha256", ""),
                name="imported D106 Phase1 tap code",
            )
        ),
        "construction_code_sha256": fixture["construction_code_sha256"],
        "integration_entry_code_sha256": _sha256_bytes(
            _read_regular(
                INTEGRATION_CODE_PATH,
                expected_sha256=runtime.get("integration_entry_code_sha256", ""),
                name="D106 integration entry code",
            )
        ),
        "model_augmentation_code_sha256": _sha256_bytes(
            _read_regular(
                MODEL_AUGMENTATION_PATH,
                expected_sha256=runtime.get("model_augmentation_code_sha256", ""),
                name="D106 model augmentation code",
            )
        ),
        "model_factory_code_sha256": _sha256_bytes(
            _read_regular(
                MODEL_FACTORY_PATH,
                expected_sha256=runtime.get("model_factory_code_sha256", ""),
                name="D106 model factory code",
            )
        ),
        "model_backbone_code_sha256": _sha256_bytes(
            _read_regular(
                MODEL_BACKBONE_PATH,
                expected_sha256=runtime.get("model_backbone_code_sha256", ""),
                name="D106 model backbone code",
            )
        ),
        "checkpoint_sha256": fixture["checkpoint_sha256"],
        "source_split_manifest_sha256": fixture["source_split_manifest_sha256"],
        "upstream_source_pool_cache_set_sha256": fixture[
            "upstream_source_pool_cache_set_sha256"
        ],
        "storage_validator_schema": LS_IQ_VALIDATOR_SCHEMA,
        "source_held_truth_access": False,
        "formal_query_access": False,
        "target_access": False,
        "performance_metrics_computed": False,
    }
    if (
        type(runtime) is not dict
        or set(runtime) != RUNTIME_FIELDS
        or bound_payloads["runtime_manifest"] != _canonical_bytes(runtime) + b"\n"
        or runtime != expected_runtime
    ):
        raise D106RealIntegrationError("D106 runtime manifest semantic closure drift")
    return fixture, _sha256_bytes(payload)


def _write_new(path: Path, value: bytes) -> None:
    if path.is_symlink():
        raise D106RealIntegrationError(f"refusing symlink output: {path}")
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def run_real_integration(
    *,
    fixture_path: str | Path,
    output_dir: str | Path,
    device: str,
) -> Mapping[str, Any]:
    fixture, fixture_sha256 = load_fixture(fixture_path)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise D106RealIntegrationError("output parent must be an existing directory")
    output.mkdir()

    extracted = extract_d106_ls_received_iq(
        source_split_manifest=fixture["source_split_manifest"],
        source_split_manifest_sha256=fixture["source_split_manifest_sha256"],
        disjoint_receipt=fixture["disjoint_receipt"],
        disjoint_receipt_sha256=fixture["disjoint_receipt_sha256"],
        upstream_source_pool_cache_set=fixture["upstream_source_pool_cache_set"],
        selection_salt_receipt=fixture["selection_salt_receipt"],
        output_dir=output / "selected_ls_iq",
    )
    tap = export_d106_phase1_ls_tap(
        selected_iq_archive=extracted["archive"],
        selected_iq_archive_sha256=extracted["archive_sha256"],
        selected_iq_receipt=extracted["receipt"],
        selected_iq_receipt_sha256=extracted["receipt_sha256"],
        storage_validator_receipt=extracted["validator_receipt"],
        storage_validator_receipt_sha256=extracted["validator_receipt_sha256"],
        ls_archive=fixture["ls_archive"],
        ls_archive_sha256=fixture["ls_archive_sha256"],
        checkpoint=fixture["checkpoint"],
        checkpoint_sha256=fixture["checkpoint_sha256"],
        runtime_manifest=fixture["runtime_manifest"],
        runtime_sha256=fixture["runtime_sha256"],
        output_dir=output / "strict_tap",
        device=device,
    )
    loaded_tap = load_d106_phase1_ls_tap(
        tap["archive"],
        tap["receipt"],
        expected_archive_sha256=tap["archive_sha256"],
        expected_receipt_sha256=tap["receipt_sha256"],
    )
    if len(loaded_tap.physical_ids) != EXPECTED_COUNTS["L_s"]:
        raise D106RealIntegrationError("formal tap row-count closure drift")

    build_lock = D106RDCEBuildLock(
        method_lock_sha256=fixture["method_lock_sha256"],
        construction_code_sha256=fixture["construction_code_sha256"],
    )
    asset = build_d106_rdce_asset(
        tap["archive"],
        tap["receipt"],
        expected_tap_archive_sha256=tap["archive_sha256"],
        expected_tap_receipt_sha256=tap["receipt_sha256"],
        build_lock=build_lock,
    )
    saved = save_d106_rdce_asset(asset, output / "rdce_asset")
    reloaded = load_d106_rdce_asset(
        output / "rdce_asset",
        expected_wire_sha256=saved["wire_sha256"],
        expected_lineage=asset.lineage,
    )
    if (
        reloaded.asset_receipt_sha256 != asset.asset_receipt_sha256
        or reloaded.binding_sha256 != asset.binding_sha256
    ):
        raise D106RealIntegrationError("RDCE wire roundtrip binding drift")

    result = {
        "schema": RESULT_SCHEMA,
        "status": "D106_REAL_INTEGRATION_COMPLETE_NO_QUERY",
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "release_commit": fixture["release_commit"],
        "fixture_sha256": fixture_sha256,
        "method_lock_sha256": fixture["method_lock_sha256"],
        "construction_code_sha256": fixture["construction_code_sha256"],
        "source_split_manifest_sha256": fixture["source_split_manifest_sha256"],
        "disjoint_receipt_sha256": fixture["disjoint_receipt_sha256"],
        "upstream_source_pool_cache_set_sha256": fixture[
            "upstream_source_pool_cache_set_sha256"
        ],
        "selected_iq_archive_sha256": extracted["archive_sha256"],
        "selected_iq_receipt_sha256": extracted["receipt_sha256"],
        "storage_validator_receipt_sha256": extracted["validator_receipt_sha256"],
        "tap_archive_sha256": tap["archive_sha256"],
        "tap_receipt_sha256": tap["receipt_sha256"],
        "rdce_wire_sha256": saved["wire_sha256"],
        "rdce_asset_receipt_sha256": asset.asset_receipt_sha256,
        "rdce_binding_sha256": asset.binding_sha256,
        "selected_row_count": EXPECTED_COUNTS["L_s"],
        "rdce_rank": asset.rank,
        "device": str(device),
        "source_held_truth_access": False,
        "formal_query_access": False,
        "target_access": False,
        "performance_metrics_computed": False,
    }
    result_path = output / RESULT_NAME
    _write_new(result_path, _canonical_bytes(result))
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "result_name": RESULT_NAME,
        "result_sha256": _sha256_bytes(result_path.read_bytes()),
        "required_directories": ["selected_ls_iq", "strict_tap", "rdce_asset"],
        "partial_output_acceptable": False,
    }
    _write_new(output / COMPLETION_NAME, _canonical_bytes(completion))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen D106 DATA-to-RDCE no-query real integration"
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_real_integration(
        fixture_path=args.fixture,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
