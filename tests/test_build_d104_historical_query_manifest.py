from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "code" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
try:
    import build_d104_historical_query_manifest as module
finally:
    sys.path.remove(str(SCRIPT_ROOT))


def _synthetic(monkeypatch):
    query = [f"q{index:03d}" for index in range(8)]
    support = [f"s{index:03d}" for index in range(42)]
    packages = [
        {
            "held_receiver": f"r{index}",
            "K": 1,
            "support_count": 6,
            "query_count": index + 1,
            "query_physical_id_root_sha256": f"q{index}",
            "support_physical_id_root_sha256": f"s{index}",
        }
        for index in range(7)
    ]
    monkeypatch.setattr(module, "HISTORICAL_QUERY_COUNT", len(query))
    monkeypatch.setattr(
        module,
        "HISTORICAL_QUERY_CANONICAL_ROOT_SHA256",
        module.canonical_sha256(query),
    )
    monkeypatch.setattr(module, "EXPECTED_SUPPORT_ROOT_SHA256", module.canonical_sha256(support))
    monkeypatch.setattr(
        module,
        "EXPECTED_PACKAGES",
        tuple(
            (
                row["held_receiver"],
                row["query_count"],
                row["query_physical_id_root_sha256"],
                row["support_physical_id_root_sha256"],
            )
            for row in packages
        ),
    )
    return query, support, packages


def test_frozen_file_identity_rejects_input_or_code_drift(monkeypatch, tmp_path) -> None:
    paths = {
        "source_partition": tmp_path / "source.py",
        "held_packager": tmp_path / "held.py",
    }
    tap = tmp_path / "tap.npz"
    dual = tmp_path / "dual.npz"
    expected = {
        tap: "a" * 64,
        dual: "b" * 64,
        paths["source_partition"]: "c" * 64,
        paths["held_packager"]: "d" * 64,
    }
    monkeypatch.setattr(module, "_sha256", lambda path: expected[path])
    monkeypatch.setattr(module, "EXPECTED_TAP_SHA256", "a" * 64)
    monkeypatch.setattr(module, "EXPECTED_DUAL_SHA256", "b" * 64)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_PARTITION_CODE_SHA256", "c" * 64)
    monkeypatch.setattr(module, "EXPECTED_HELD_PACKAGER_CODE_SHA256", "d" * 64)
    module._validate_frozen_files(tap, dual, paths)
    expected[dual] = "0" * 64
    with pytest.raises(ValueError, match="identity drift"):
        module._validate_frozen_files(tap, dual, paths)


def test_source_registry_and_root_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(module, "EXPECTED_RECEIVERS", ("r0",))
    monkeypatch.setattr(module, "EXPECTED_CLASSES", ("c0",))
    monkeypatch.setattr(module, "EXPECTED_SOURCE_VAL_ROOT_SHA256", "a" * 64)
    receipt = {
        "counts": {"L_s": 588, "U_s": 5292, "source_val": 2520},
        "physical_id_roots": {"source_val": "a" * 64},
    }
    module._validate_source_registry(receipt, ("r0",), ("c0",))
    changed = copy.deepcopy(receipt)
    changed["physical_id_roots"]["source_val"] = "b" * 64
    with pytest.raises(ValueError, match="registry/root drift"):
        module._validate_source_registry(changed, ("r0",), ("c0",))


@pytest.mark.parametrize(
    "mutation",
    ("query_add", "query_duplicate", "support_duplicate", "intersection", "package"),
)
def test_reconstruction_negatives(monkeypatch, mutation) -> None:
    query, support, packages = _synthetic(monkeypatch)
    changed_query = list(query)
    changed_support = list(support)
    changed_packages = copy.deepcopy(packages)
    if mutation == "query_add":
        changed_query.append("extra")
    elif mutation == "query_duplicate":
        changed_query[-1] = changed_query[0]
    elif mutation == "support_duplicate":
        changed_support[-1] = changed_support[0]
    elif mutation == "intersection":
        changed_support[-1] = changed_query[0]
    else:
        changed_packages[-1]["query_count"] += 1
    with pytest.raises(ValueError, match="reconstruction drift"):
        module._validate_reconstruction(
            changed_query, changed_support, changed_packages
        )


def test_exclusive_manifest_write_rejects_existing_output(tmp_path) -> None:
    output = tmp_path / "manifest.json"
    module._write_json_new(output, {"schema": "test"})
    with pytest.raises(FileExistsError):
        module._write_json_new(output, {"schema": "changed"})
