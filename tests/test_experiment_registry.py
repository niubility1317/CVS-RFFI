import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/experiment_registry.py"
spec = importlib.util.spec_from_file_location("experiment_registry", MODULE_PATH)
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def valid_spec():
    s = registry.template()
    s.update(run_id="20260916-comparison-method-m2-r01", description="两个模型seed共享同一数据划分", kind="comparison")
    r = s["rows"][0]
    r.update(row_id="arm-s000001", method="MoPC", output_root="/runs/run/arm-s1", seeds={k: 5 for k in registry.SEED_ROLES})
    r["seeds"]["model"] = 1
    second = json.loads(json.dumps(r))
    second.update(row_id="arm-s000002", output_root="/runs/run/arm-s2")
    second["seeds"]["model"] = 2
    s["rows"].append(second)
    return s


def test_registration_preserves_separate_seed_roles_and_never_overwrites(tmp_path):
    s = valid_spec()
    source = tmp_path / "spec.json"
    write(source, s)
    folder = registry.register(tmp_path, source)
    assert registry.read_json(folder / "experiment.json")["rows"][1]["seeds"] == {**{k: 5 for k in registry.SEED_ROLES}, "model": 2}
    report_before = (folder / "report.md").read_bytes()
    with pytest.raises(FileExistsError):
        registry.register(tmp_path, source)
    assert (folder / "report.md").read_bytes() == report_before
    registry.build(tmp_path, managed_only=True)
    rows, count = registry.search(tmp_path, seed=2, kind="comparison")
    assert count == 1 and len(rows[0]["rows"]) == 2


def test_row_state_does_not_complete_whole_run_and_events_survive_rebuild(tmp_path):
    source = tmp_path / "spec.json"
    s = valid_spec()
    write(source, s)
    folder = registry.register(tmp_path, source)
    registry.event(tmp_path, s["run_id"], "RUNNING", ["readback.json"], "核实PID与log")
    registry.event(tmp_path, s["run_id"], "ANALYZED", ["score.json"], "仅第一行完成", "arm-s000001")
    registry.build(tmp_path, managed_only=True)
    row = next(registry.catalog_rows(tmp_path))
    assert row["status"] == "RUNNING"
    assert row["row_states"]["arm-s000001"]["status"] == "ANALYZED"
    assert row["row_states"]["arm-s000001"]["evidence"] == ["score.json"]
    assert row["events"].endswith("events.jsonl")
    assert len((folder / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    with pytest.raises(ValueError):
        registry.event(tmp_path, s["run_id"], "FAILED", ["err.log"], "wrong row", "not-a-row")


def test_bad_ids_duplicate_outputs_and_missing_roles_rejected():
    s = valid_spec()
    s["run_id"] = "../escape"
    s["rows"][1]["output_root"] = s["rows"][0]["output_root"] + "/"
    s["rows"][0]["seeds"]["model"] = True
    errors = registry.validate_spec(s)
    assert any("ASCII" in e for e in errors)
    assert any("重复output_root" in e for e in errors)
    assert any("seed model" in e for e in errors)


def test_legacy_same_names_keep_paths_scoped_values_and_historical_status(tmp_path):
    for surface, seed in [("automation_reports/CV-SincNet", 1), ("runs", 2)]:
        folder = tmp_path / surface / "same-name"
        write(folder / "config.json", {"rows": [{"row_id": "first", "model_seed": seed, "split_seed": 9, "lr": 0.1},
                                                {"row_id": "second", "model_seed": seed + 10, "split_seed": 9, "lr": 0.2}]})
        write(folder / "status.json", {"status": "RUNNING"})
        (folder / "report.md").write_text("# 中文报告\n说明历史RUNNING不代表现在", encoding="utf-8")
        (folder / "weights.pth").write_bytes(b"\xff\x00binary-do-not-read")
    coverage = registry.build(tmp_path)
    rows, count = registry.search(tmp_path, "same-name")
    assert count == 2
    assert all(r["status"] == "HISTORICAL_UNVERIFIED" for r in rows)
    assert all("RUNNING" in r["reported_statuses"] for r in rows)
    with pytest.raises(ValueError, match="匹配2项"):
        registry.show(tmp_path, "same-name")
    fact_rows = registry.show(tmp_path, rows[0]["id"], "facts")["shown"]
    model_seeds = [f for f in fact_rows if f["field"] == "model_seed"]
    assert {f["row"] for f in model_seeds} == {"first", "second"}
    assert all(f["source"].endswith("config.json") for f in model_seeds)
    assert not coverage["errors"]
    prior = {r["id"] for r in rows}
    registry.build(tmp_path, managed_only=True)
    assert {r["id"] for r in registry.catalog_rows(tmp_path)} == prior


def test_missing_directory_is_disclosed_and_old_locator_retained(tmp_path):
    p = tmp_path / "runs/old/config.json"
    write(p, {"seed": 123})
    registry.build(tmp_path)
    p.rename(p.with_suffix(".unrecognized"))
    coverage = registry.build(tmp_path)
    rows = list(registry.catalog_rows(tmp_path))
    assert rows[0]["evidence_state"] == "NOT_SEEN_IN_LATEST_SCAN_USE_PREVIOUS_LOCATOR"
    assert any(not s["exists"] for s in coverage["surfaces"])


def test_template_is_not_launch_ready_and_metadata_does_not_copy_truth():
    assert registry.validate_spec(registry.template(), ready=True)
    extracted = list(registry.facts({"seed": 8, "predictions": [{"seed": 777, "truth": 4}], "physical_ids": list(range(500))}))
    assert extracted == [{"field": "seed", "value": 8, "scope": "$.seed", "row": None}]


def test_backfill_without_events_keeps_report_and_is_unknown(tmp_path):
    s = valid_spec()
    folder = registry.record_dir(tmp_path, s["run_id"])
    write(folder / "experiment.json", s)
    original = "# 历史报告\n完成状态尚未核实"
    (folder / "report.md").write_text(original, encoding="utf-8")
    registry.build(tmp_path, managed_only=True)
    rec = next(registry.catalog_rows(tmp_path))
    assert rec["status"] == "UNKNOWN"
    assert (folder / "report.md").read_text(encoding="utf-8") == original


def test_plain_json_filename_and_utf16_metadata_and_log_catalog_import(tmp_path):
    cfg = tmp_path / "code/configs/arm_s123.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"model_seed": 123, "split_seed": 9}), encoding="utf-16")
    csv_path = tmp_path / registry.LOG_CATALOGS[0]
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("source_path,model_key,seed,training_route,status,dataset\nold.log,DRIFT-A,42,comparison_baseline,finished,wisig\n", encoding="utf-8")
    coverage = registry.build(tmp_path)
    assert not coverage["errors"]
    assert registry.search(tmp_path, seed=123)[1] == 1
    matches, count = registry.search(tmp_path, "DRIFT", seed=42)
    assert count == 1 and matches[0]["record_type"] == "legacy_log_record"
    assert registry.show(tmp_path, matches[0]["id"], "facts")["total"] >= 4


def test_remote_snapshot_is_a_locator_and_does_not_claim_running(tmp_path):
    snap = {"schema": "n607_experiment_directory_inventory_v1", "observed_at": "2026-09-16T00:00:00+00:00",
            "host": "test-n607", "depth": "surface/group/immediate_children", "errors": [],
            "records": [{"path": "/remote/runs/daot-test", "name": "daot-test", "surface": "runs",
                         "entry_type": "directory", "children": [{"name": "row-a", "type": "directory"}]}]}
    write(tmp_path / "experiment_registry/snapshots/test.json", snap)
    result = registry.build(tmp_path, managed_only=True)
    rec = next(registry.catalog_rows(tmp_path))
    assert rec["record_type"] == "remote_directory_locator" and rec["status"] == "UNKNOWN"
    assert registry.show(tmp_path, rec["id"])["children"][0]["name"] == "row-a"
    assert result["remote_snapshots"][0]["records"] == 1
