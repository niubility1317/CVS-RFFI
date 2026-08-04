from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "code" / "scripts" / "build_next_r5_fa_target125_asset.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("test_next_r5_target125_asset_builder", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_tap_arrays(module: object) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(713102)
    receivers = ("1-1", "18-2", "1-19", "2-1", "2-19", "14-7", "19-2")
    days = ("2021_03_01", "2021_03_08", "2021_03_15", "2021_03_23")
    classes = ("6-15", "8-20", "14-7", "14-10", "20-15", "20-19")
    per_day = (4, 4, 3, 3)
    values: dict[str, list[object]] = {
        "pre_relu": [],
        "z_dom": [],
        "tx_labels": [],
        "receiver_ids": [],
        "day_ids": [],
        "physical_ids": [],
        "scenario_names": [],
        "observation_ids": [],
    }
    for receiver_index, receiver in enumerate(receivers):
        for class_index, class_handle in enumerate(classes):
            for day_index, (day, count) in enumerate(zip(days, per_day, strict=True)):
                for sample in range(count):
                    row = rng.uniform(0.01, 0.05, size=160)
                    row[class_index * 5 : class_index * 5 + 5] += 0.8
                    row[48 + receiver_index * 4 : 48 + receiver_index * 4 + 4] += 0.3
                    row[92 + day_index * 4 : 92 + day_index * 4 + 4] += 0.2
                    row[120 + (receiver_index + day_index) % 10] += 0.15
                    row = np.maximum(row + rng.normal(0.0, 0.002, size=160), 1.0e-4).astype(np.float32)
                    values["pre_relu"].append(row)
                    values["z_dom"].append((row * np.float32(0.5)).astype(np.float32))
                    values["tx_labels"].append(class_handle)
                    values["receiver_ids"].append(receiver)
                    values["day_ids"].append(day)
                    values["physical_ids"].append(f"pid-secret-{receiver}-{day}-{class_handle}-{sample}")
                    values["scenario_names"].append("leo_clear_weak")
                    values["observation_ids"].append(f"obs-{receiver}-{day}-{class_handle}-{sample}")
    arrays = {
        "pre_relu": np.asarray(values["pre_relu"], dtype=np.float32),
        "z_dom": np.asarray(values["z_dom"], dtype=np.float32),
        "tx_labels": np.asarray(values["tx_labels"], dtype="<U16"),
        "receiver_ids": np.asarray(values["receiver_ids"], dtype="<U16"),
        "day_ids": np.asarray(values["day_ids"], dtype="<U16"),
        "physical_ids": np.asarray(values["physical_ids"], dtype="<U64"),
        "scenario_names": np.asarray(values["scenario_names"], dtype="<U32"),
        "observation_ids": np.asarray(values["observation_ids"], dtype="<U96"),
    }
    assert tuple(arrays) == tuple(module.STRICT_TAP_MEMBERS)
    assert len(arrays["pre_relu"]) == 588
    return arrays


def test_direct_strict_tap_cli_builds_one_six_old_class_source_only_asset(tmp_path: Path) -> None:
    module = _module()
    tap = (tmp_path / "d106_strict_tap.npz").resolve()
    np.savez(tap, **_strict_tap_arrays(module))
    output = (tmp_path / "target125_fa_asset").resolve()
    result = module.build_target125_fa_asset(
        strict_tap=tap,
        strict_tap_sha256=_sha(tap.read_bytes()),
        checkpoint_sha256=_sha(b"checkpoint"),
        method_lock_sha256=_sha(b"method-lock"),
        output_dir=output,
    )
    assert result["status"] == module.BUILD_STATUS
    assert result["old_class_count"] == 6
    assert result["target_support_rows_used"] == 0
    manifest_path = Path(str(result["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase1_source_only"] is True
    assert manifest["strict_tap_sha256"] == _sha(tap.read_bytes())
    assert manifest["phase1_source_rows_retained"] is False
    wire = Path(str(result["asset"])).read_bytes()
    assert "pid-secret-" not in wire.decode("ascii")
    assert "pid-secret-" not in manifest_path.read_text(encoding="utf-8")
    asset = module.core.deserialize_target_fa_asset(wire)
    assert len(asset.old_classes) == 6
    assert asset.fa_asset.aggregate_samples_per_class == (98,) * 6


def test_cli_accepts_only_strict_tap_lineage_inputs() -> None:
    module = _module()
    names = {action.dest for action in module._parser()._actions}
    assert {
        "strict_tap",
        "strict_tap_sha256",
        "checkpoint_sha256",
        "method_lock_sha256",
        "output_dir",
    }.issubset(names)
    assert not {"target", "support", "query", "truth", "rank", "rho", "kappa"}.intersection(names)
