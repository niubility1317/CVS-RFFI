import json
from pathlib import Path
import sys
import pytest

from comparison_suite.launch import completed, eligible, launch, row_key, fresh_rows_preflight, release_preflight


def test_lane_and_foreign_gpu_budget_and_dependency(tmp_path):
    row = dict(stage="phase12", row_id="x", gpu=0, output_root=str(tmp_path / "output"), dependency=str(tmp_path / "completion.json"))
    state = {row_key(row): {"status": "QUEUED"}}
    assert not eligible(row, state, {}, {})
    (tmp_path / "completion.json").write_text(json.dumps({"status": "SOURCE_TRAINED"}))
    (tmp_path / "last.pt").write_bytes(b"synthetic")
    assert not eligible(row, state, {}, {})
    state["source:x"] = {"status": "TRAINING_COMPLETE"}
    assert eligible(row, state, {}, {0: {22}})
    assert not eligible(row, state, {}, {0: {22, 23}})
    class P:
        pid = 1
    active = {"x": (row, P(), None)}
    assert not eligible(row, state, active, {})


def test_artifact_readback_rejects_exit_only(tmp_path):
    row = dict(stage="source", output_root=str(tmp_path))
    (tmp_path / "completion.json").write_text(json.dumps({"status": "SOURCE_TRAINED"}))
    assert not completed(row)
    (tmp_path / "last.pt").write_bytes(b"test-artifact")
    assert completed(row)


def test_no_retry_and_dependency_failure_actual_subprocess(tmp_path):
    stub = tmp_path / "worker.py"
    stub.write_text('''import sys,json
from pathlib import Path
c=json.loads(Path(sys.argv[-1]).read_text())
if c.get("fail"): raise SystemExit(3)
p=Path(c["out"]);p.mkdir(exist_ok=True)
(p/"last.pt").write_bytes(b"synthetic launcher test only")
(p/"completion.json").write_text(json.dumps({"status":"SOURCE_TRAINED"}))
''')
    good_out, bad_out = tmp_path / "good", tmp_path / "bad"
    rows = []
    for name, output, fail in (("good",good_out,False),("bad",bad_out,True)):
        cfg = tmp_path / (name + ".json")
        cfg.write_text(json.dumps({"out":str(output),"fail":fail}))
        rows.append(dict(stage="source",row_id=name,run_id="synthetic",method="stub",model_seed=0,
            config=str(cfg),module="worker",gpu=0,output_root=str(output),log=str(output / "log"),dependency=None))
    dep = dict(stage="phase12",row_id="bad",run_id="synthetic",method="stub",model_seed=0,
        config=str(tmp_path / "unused.json"),module="worker",gpu=0,output_root=str(tmp_path / "blocked"),
        log=str(tmp_path / "blocked/log"),dependency=str(bad_out / "completion.json"))
    rows.append(dep)
    spec = dict(run_id="synthetic",source_run_id="synthetic",rows=rows,root=str(tmp_path),code_root=str(tmp_path),
        python=sys.executable,runtime_root=str(tmp_path / "runtime"),launch_owner="pytest-only",cpu_test_only=True,poll_seconds=.01)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    states = launch(path)
    assert states["source:good"]["status"] == "TRAINING_COMPLETE"
    assert states["source:bad"]["status"] == "FAILED"
    assert states["phase12:bad"]["status"] == "FAILED"
    assert not (tmp_path / "blocked").exists()
    with pytest.raises(FileExistsError):
        launch(path)


def test_source_artifacts_cannot_be_reused_under_fresh_dispatcher(tmp_path):
    out = tmp_path / "source"
    out.mkdir()
    (out / "last.pt").write_bytes(b"old")
    row = dict(stage="source", row_id="method-s0", output_root=str(out), log=str(out / "log"), dependency=None)
    with pytest.raises(FileExistsError, match="reconcile"):
        fresh_rows_preflight({"rows": [row]})


def test_release_oid_and_cwd_verified(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    oid = "a" * 40
    (tmp_path / "release_commit.txt").write_text(oid)
    release_preflight({"code_root": str(tmp_path)}, oid)
    with pytest.raises(ValueError, match="differs"):
        release_preflight({"code_root": str(tmp_path)}, "b" * 40)
    with pytest.raises(ValueError, match="cwd"):
        release_preflight({"code_root": str(tmp_path / "other")}, oid)


def test_matrix_exact_45_distinct_source_and_pipeline_rows(monkeypatch, tmp_path):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/prepare_native_comparison_matrix.py"
    spec = importlib.util.spec_from_file_location("matrix_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: "test-commit")
    registry = Path("E:/type10-7/tools/experiment_registry.py")
    if not registry.exists():
        registry = root / "tools/experiment_registry.py"
    result = mod.prepare(tmp_path, "/remote", registry)
    sources = [r for r in result["rows"] if r["stage"] == "source"]
    targets = [r for r in result["rows"] if r["stage"] == "phase12"]
    assert len(sources) == len(targets) == 45
    assert len({r["output_root"] for r in result["rows"]}) == 90
    assert {r["method"] for r in sources} == set(mod.METHODS)
    radio = json.loads((tmp_path / "automation_reports/CV-SincNet" / mod.SOURCE_RUN / "configs/radionet_ada-s392005.json").read_text())
    assert radio["epochs"] == 200 and radio["fs_hz"] == 25e6
    assert radio["split_seed"] == 392005 and radio["augmentation_seed"] == 2027
    phase = json.loads((tmp_path / "automation_reports/CV-SincNet" / mod.PIPELINE_RUN / "configs/csil-s392005.json").read_text())
    assert phase["registration"]["csil_old_fingerprint_trainable"]
    assert "version" not in phase["registration"]
    with pytest.raises(FileExistsError):
        mod.prepare(tmp_path, "/remote", registry)
