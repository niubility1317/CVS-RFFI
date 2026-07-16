from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from paper_reproduction.scripts.build_adv3b02_ci_strict_plan import build
from paper_reproduction.scripts.run_adv3b02_ci_strict_plan import (
    _load_formal_rows,
    _load_plan,
)
from paper_reproduction.scripts import run_adv3b02_ci_strict_plan as runner


def _args(tmp_path: Path, split_path: Path, *, smoke_receipt: Path | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ("base_runtime", "candidate_lock", "adapter", "head_artifact", "tta_policy"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts[name] = path
    return argparse.Namespace(
        experiment_id="test_ci",
        run_root=str(tmp_path / "remote_run"),
        target_cache_root=str(tmp_path / "target_cache"),
        class_split=split_path,
        smoke_receipt=smoke_receipt,
        output=tmp_path / ("authorized.json" if smoke_receipt else "plan.json"),
        **{key: str(value) for key, value in artifacts.items()},
    )


def _split(tmp_path: Path) -> Path:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "target_old_tx_labels": [f"old-{i}" for i in range(6)],
        "nested_target_new_tx_labels": {
            "5": [f"new-{i}" for i in range(5)],
            "10": [f"new-{i}" for i in range(10)],
            "20": [f"new-{i}" for i in range(20)],
        },
        "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "confirmation_seeds": [713101, 713102, 713103, 713104, 713105],
        "k_values": [1, 5, 10, 20],
    }), encoding="utf-8")
    return path


def test_plan_has_exact_75_packages_900_cells_and_no_initial_authority(tmp_path: Path):
    args = _args(tmp_path, _split(tmp_path))
    plan = build(args)
    assert plan["counts"] == {"packages": 75, "cells": 900, "scenario_rows": 2700}
    assert plan["launch_authority"] is False
    assert plan["phase2_query_role_oracle_access"] is False
    assert plan["phase2_query_class_quota_access"] is False
    assert _load_plan(args.output)["smoke_cell_id"].endswith("__csil__k_1")


def test_only_exact_pass_smoke_authorizes_matrix(tmp_path: Path):
    split = _split(tmp_path)
    bad = tmp_path / "bad_smoke.json"
    bad.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    with pytest.raises(ValueError):
        build(_args(tmp_path / "bad", split, smoke_receipt=bad))

    good_root = tmp_path / "good"
    good_root.mkdir()
    good = good_root / "smoke.json"
    good.write_text(json.dumps({
        "schema": "cvs.phase2.adv3b02_ci_smoke_receipt.v1",
        "status": "PASS",
        "package_id": "rx_20_1__seed_713101__new_5",
        "method": "csil",
        "k_shot": 1,
    }), encoding="utf-8")
    plan = build(_args(good_root / "inputs", split, smoke_receipt=good))
    assert plan["launch_authority"] is True
    assert plan["authority_state"] == "N607_CI_SMOKE_PASS"


def test_matrix_sharding_is_package_exclusive():
    source = __import__("inspect").getsource(runner.run)
    assert 'for index, package in enumerate(plan["packages"])' in source
    assert 'cells_by_package[package["package_id"]]' in source


def test_ci_runner_uses_isolated_structural_bundle_builder():
    source = __import__("inspect").getsource(runner._build_package)
    assert "build_adv3b02_ci_predictor_bundle.py" in source
    assert "build_cvs_stage2_predictor_bundle.py" not in source


def test_formal_rows_loader_requires_schema_wrapped_three_rows(tmp_path: Path):
    path = tmp_path / "formal_rows.json"
    path.write_text(json.dumps({
        "schema": "cvs.phase2.formal_metric_rows.v1",
        "rows": [{"scenario": value} for value in (
            "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"
        )],
    }), encoding="utf-8")
    assert len(_load_formal_rows(path)) == 3

    path.write_text(json.dumps([{"scenario": "leo_clear_weak"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_formal_rows(path)
