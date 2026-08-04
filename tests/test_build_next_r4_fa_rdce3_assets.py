from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_next_r4_fa_rdce3 as fa
from cvsrffi import stage2_next_r4_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "code" / "scripts" / "build_next_r4_fa_rdce3_assets.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("test_next_r4_fa_asset_builder", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_tap_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(713102)
    receivers = ("1-1", "18-2", "1-19", "2-1", "2-19", "14-7", "19-2")
    days = ("2021_03_01", "2021_03_08", "2021_03_15", "2021_03_23")
    classes = ("6-15", "8-20", "14-7", "14-10", "20-15", "20-19")
    count_by_day = (4, 4, 3, 3)
    pre_rows: list[np.ndarray] = []
    zdom_rows: list[np.ndarray] = []
    receiver_ids: list[str] = []
    day_ids: list[str] = []
    tx_labels: list[str] = []
    physical_ids: list[str] = []
    observation_ids: list[str] = []
    scenario_names: list[str] = []
    for receiver_index, receiver in enumerate(receivers):
        for class_index, class_handle in enumerate(classes):
            for day_index, (day, count) in enumerate(zip(days, count_by_day, strict=True)):
                for sample in range(count):
                    # Positive pre-ReLU rows make the R0 conversion visible while
                    # retaining independent class and receiver-day directions.
                    value = rng.uniform(0.01, 0.05, size=fa.Z_DIM)
                    value[class_index * 5 : class_index * 5 + 5] += 0.8
                    value[48 + receiver_index * 4 : 48 + receiver_index * 4 + 4] += 0.3
                    value[92 + day_index * 4 : 92 + day_index * 4 + 4] += 0.2
                    value[120 + (receiver_index + day_index) % 10] += 0.15
                    value += rng.normal(0.0, 0.002, size=fa.Z_DIM)
                    value = np.maximum(value, 1.0e-4).astype(np.float32)
                    pre_rows.append(value)
                    zdom_rows.append((value * np.float32(0.5)).astype(np.float32))
                    receiver_ids.append(receiver)
                    day_ids.append(day)
                    tx_labels.append(class_handle)
                    physical_ids.append(f"pid-secret-{receiver}-{day}-{class_handle}-{sample}")
                    observation_ids.append(f"observation-{receiver}-{day}-{class_handle}-{sample}")
                    scenario_names.append("leo_clear_weak")
    assert len(pre_rows) == 588
    return {
        "pre_relu": np.asarray(pre_rows, dtype=np.float32),
        "z_dom": np.asarray(zdom_rows, dtype=np.float32),
        "tx_labels": np.asarray(tx_labels, dtype="<U16"),
        "receiver_ids": np.asarray(receiver_ids, dtype="<U16"),
        "day_ids": np.asarray(day_ids, dtype="<U16"),
        "physical_ids": np.asarray(physical_ids, dtype="<U64"),
        "scenario_names": np.asarray(scenario_names, dtype="<U32"),
        "observation_ids": np.asarray(observation_ids, dtype="<U96"),
    }


def _write_tap(path: Path) -> tuple[dict[str, np.ndarray], str]:
    arrays = _strict_tap_arrays()
    np.savez(path, **arrays)
    return arrays, _sha(path.read_bytes())


def test_builds_twelve_aggregate_only_assets_from_full_strict_tap(tmp_path: Path) -> None:
    module = _module()
    tap_path = (tmp_path / "d106_ls_strict_tap.npz").resolve()
    arrays, tap_sha = _write_tap(tap_path)
    output = (tmp_path / "fa_assets").resolve()
    checkpoint_sha = _sha(b"checkpoint")
    lock_sha = _sha(b"method-lock")

    result = module.build_next_r4_fa_rdce3_assets(
        strict_tap=tap_path,
        strict_tap_sha256=tap_sha,
        checkpoint_sha256=checkpoint_sha,
        method_lock_sha256=lock_sha,
        output_dir=output,
    )

    assert result["status"] == module.BUILD_STATUS
    assert result["asset_count"] == 12
    assert result["outer_phase1_fit_count"] == 420
    assert result["old_classes_per_asset"] == 5
    assert result["phase1_member_ids_written"] is False
    assert result["phase1_per_row_features_written"] is False
    manifest_path = Path(str(result["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = tuple(sorted(set(arrays["tx_labels"].astype(str).tolist())))
    assert manifest["schema"] == module.ASSET_MANIFEST_SCHEMA
    assert set(manifest["entries"]) == {
        f"{receiver}|{class_handle}"
        for receiver in matrix.HELD_RECEIVERS
        for class_handle in classes
    }
    for key, entry in manifest["entries"].items():
        assert set(entry) == {
            "asset_path",
            "asset_sha256",
            "checkpoint_sha256",
            "phase1_fit_physical_root_sha256",
        }
        assert entry["checkpoint_sha256"] == checkpoint_sha
        wire = Path(str(entry["asset_path"])).read_bytes()
        assert _sha(wire) == entry["asset_sha256"]
        # The input-only physical IDs must not cross the aggregate wire or manifest boundary.
        assert "pid-secret-" not in wire.decode("ascii")
        assert "pid-secret-" not in manifest_path.read_text(encoding="utf-8")
        asset = fa.deserialize_fa_rdce3_phase1_asset(wire)
        assert asset.checkpoint_sha256 == checkpoint_sha
        assert asset.phase1_bundle_sha256 == tap_sha
        assert asset.method_lock_sha256 == lock_sha
        assert asset.aggregate_samples_per_class == (84, 84, 84, 84, 84)
        assert len(asset.old_classes) == 5
        assert np.all(fa.decode_fa_rdce3_fisher_precision(asset) > 0.0)
        assert np.all(fa.decode_fa_rdce3_residual_variance(asset) > 0.0)
        assert 0.0 < fa.decode_fa_rdce3_radius(asset)
        assert np.all((fa.decode_fa_rdce3_kappa(asset) >= 0.0) & (fa.decode_fa_rdce3_kappa(asset) < 1.0))
        held_receiver, held_class = key.split("|", maxsplit=1)
        expected_ids = tuple(
            physical_id
            for receiver, class_handle, physical_id in zip(
                arrays["receiver_ids"].astype(str),
                arrays["tx_labels"].astype(str),
                arrays["physical_ids"].astype(str),
                strict=True,
            )
            if receiver != held_receiver and class_handle != held_class
        )
        assert entry["phase1_fit_physical_root_sha256"] == module._physical_root(expected_ids)


def test_fails_closed_before_creating_output_for_bad_grid(tmp_path: Path) -> None:
    module = _module()
    tap_path = (tmp_path / "bad_strict_tap.npz").resolve()
    arrays = _strict_tap_arrays()
    arrays["pre_relu"] = arrays["pre_relu"][:-1]
    np.savez(tap_path, **arrays)
    output = (tmp_path / "must_not_exist").resolve()

    with pytest.raises(module.NextR4FAAssetBuildError, match="feature dtype/shape/finite"):
        module.build_next_r4_fa_rdce3_assets(
            strict_tap=tap_path,
            strict_tap_sha256=_sha(tap_path.read_bytes()),
            checkpoint_sha256=_sha(b"checkpoint"),
            method_lock_sha256=_sha(b"method-lock"),
            output_dir=output,
        )
    assert not output.exists()


def test_cli_exposes_only_frozen_phase1_inputs_and_no_target_surface() -> None:
    module = _module()
    names = {action.dest for action in module._parser()._actions}
    assert {
        "strict_tap",
        "strict_tap_sha256",
        "checkpoint_sha256",
        "method_lock_sha256",
        "output_dir",
    }.issubset(names)
    assert not {"target", "query", "truth", "seed", "kappa", "rho", "rank"}.intersection(names)
