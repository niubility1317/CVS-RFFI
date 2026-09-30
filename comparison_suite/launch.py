"""Single-owner, no-retry dependency queue; one source and one Phase12 lane/GPU."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + "." + str(os.getpid()) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def row_key(row):
    return row["stage"] + ":" + row["row_id"]


def completed(row, stage=None):
    out = Path(row["output_root"])
    flag = out / ("completion.json" if row["stage"] == "source" else "phase1_complete.json" if stage == "p1" else "phase2_complete.json")
    if not flag.exists():
        return False
    value = read_json(flag)
    expected = "SOURCE_TRAINED" if row["stage"] == "source" else "PREDICTIONS_COMPLETE"
    artifact = "last.pt" if row["stage"] == "source" else "phase1_predictions.npz" if stage == "p1" else "phase2_predictions.jsonl"
    return value.get("status") == expected and (out / artifact).is_file()


def gpu_occupancy():
    """Compute-process inventory; does not kill or change any foreign process."""
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader,nounits"], text=True)
    mapping = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in names.splitlines() if line.strip()}
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    result = {i: set() for i in mapping.values()}
    for line in processes.splitlines():
        if line.strip():
            uuid, pid = line.split(",")
            if uuid.strip() in mapping:
                result[mapping[uuid.strip()]].add(int(pid))
    return result


def eligible(row, states, active, occupancy):
    key = row_key(row)
    if states[key]["status"] != "QUEUED":
        return False
    gpu = int(row["gpu"])
    owned = [(r,p) for r,p,_ in active.values() if int(r["gpu"]) == gpu]
    if any(r["stage"] == row["stage"] for r,p in owned):
        return False
    owned_pids = {p.pid for r,p in owned}
    foreign = set(occupancy.get(gpu, ())) - owned_pids
    if len(foreign) + len(owned) >= 2:
        return False
    if row["dependency"]:
        if states.get("source:" + row["row_id"], {}).get("status") != "TRAINING_COMPLETE":
            return False
        dependency = Path(row["dependency"])
        if not dependency.exists() or read_json(dependency).get("status") != "SOURCE_TRAINED" or not (dependency.parent / "last.pt").is_file():
            return False
    return True


def release_preflight(spec, commit):
    if spec.get("cpu_test_only"):
        return
    if not commit or len(commit) != 40:
        raise ValueError("Actual 40-character committed release OID required")
    root = Path(spec["code_root"]).resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Dispatcher cwd must be the exact published code_root")
    if (root / "release_commit.txt").read_text(encoding="utf-8").strip() != commit:
        raise ValueError("Published release commit evidence differs from --commit")


def fresh_rows_preflight(spec):
    source_rows = {r["row_id"]: r for r in spec["rows"] if r["stage"] == "source"}
    for row in spec["rows"]:
        out = Path(row["output_root"])
        artifacts = [out / p for p in ("last.pt", "completion.json", "resolved_config.json", "phase1_complete.json", "phase2_complete.json", "phase1_predictions.npz", "phase2_predictions.jsonl")]
        artifacts.append(Path(row["log"]))
        if any(p.exists() for p in artifacts):
            raise FileExistsError(f"Existing row artifacts for {row_key(row)}; reconcile existing process/output before launch")
        if row["dependency"]:
            source = source_rows.get(row["row_id"])
            if source is None or Path(row["dependency"]).resolve() != (Path(source["output_root"]) / "completion.json").resolve():
                raise ValueError("Every Phase12 row must depend on its matching owned scratch source row")


def detach(spec_path, commit):
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    release_preflight(spec, commit)
    fresh_rows_preflight(spec)
    runtime = Path(spec["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    lock = runtime / "detach.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        if any((runtime / p).exists() for p in ("launch.json", "launch.lock", "state.json")):
            raise FileExistsError("Existing dispatcher evidence; reconcile before any new launch")
        command = [spec["python"], "-u", "-m", "comparison_suite.launch", "--spec", str(spec_path), "--commit", commit]
        with (runtime / "dispatcher.log").open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=spec["code_root"], stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        value = {"pid": process.pid, "argv": command, "cwd": spec["code_root"], "commit": commit,
                 "launch_owner": spec["launch_owner"], "at": datetime.now(timezone.utc).isoformat(), "status": "DISPATCHED_AWAITING_READBACK"}
        atomic_json(runtime / "launch.json", value)
        print(json.dumps(value), flush=True)
        return value
    finally:
        lock.unlink(missing_ok=True)


def launch(spec_path, commit=None):
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    release_preflight(spec, commit)
    fresh_rows_preflight(spec)
    rows = spec["rows"]
    keys = [row_key(r) for r in rows]
    if len(keys) != len(set(keys)) or len({r["output_root"] for r in rows}) != len(rows):
        raise ValueError("Each launch row must have a unique stage/row ID and output directory")
    runtime = Path(spec["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    lock = runtime / "launch.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("ascii"))
    os.close(fd)
    # A previous dispatch is never silently resumed or resubmitted.
    if (runtime / "state.json").exists():
        lock.unlink()
        raise FileExistsError("Dispatcher state exists; reconcile prior PIDs and artifacts before any new launch")
    states = {row_key(r): {"status": "QUEUED", "pid": None, "gpu": r["gpu"], "config": r["config"], "log": r["log"]} for r in rows}
    active, scorers, score_states = {}, {}, {"p1": "WAITING", "p2": "WAITING"}
    event_path = runtime / "events.jsonl"
    original_reports = {}
    for run_id in {r["run_id"] for r in rows}:
        report = Path(spec["root"]) / "automation_reports/CV-SincNet" / run_id / "report.md"
        if report.exists():
            original_reports[run_id] = report.read_text(encoding="utf-8")
    def event(status, row=None, **data):
        value = {"at": datetime.now(timezone.utc).isoformat(), "status": status, "row_id": None if row is None else row["row_id"],
                 "stage": None if row is None else row["stage"], **data}
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if row is not None:
            registry = Path(spec["root"]) / "automation_reports/CV-SincNet" / row["run_id"] / "events.jsonl"
            if registry.exists():
                with registry.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"at": value["at"], "status": status, "row_id": row["row_id"],
                        "note": data.get("note", "native dispatcher observable process/artifact transition"),
                        "evidence": [str(event_path), row["output_root"], row["log"]], **data}, ensure_ascii=False) + "\n")
    def save():
        atomic_json(runtime / "state.json", {"dispatcher_pid": os.getpid(), "launch_owner": spec["launch_owner"],
            "rows": states, "scoring": score_states, "spec": str(spec_path), "commit": commit})
        for run_id in {r["run_id"] for r in rows}:
            folder = Path(spec["root"]) / "automation_reports/CV-SincNet" / run_id
            record = folder / "experiment.json"
            if not record.exists():
                continue
            obj = read_json(record)
            run_rows = [r for r in rows if r["run_id"] == run_id]
            active_statuses = [states[row_key(r)]["status"] for r in run_rows]
            is_source = all(r["stage"] == "source" for r in run_rows)
            status = "RUNNING" if any(s in ("RUNNING", "QUEUED") for s in active_statuses) else "PARTIAL" if "FAILED" in active_statuses else "TRAINING_COMPLETE" if is_source else "ANALYZED" if score_states["p2"] == "ANALYZED" else "PARTIAL" if score_states["p2"] == "FAILED" else "PREDICTIONS_COMPLETE"
            obj["status"] = status
            obj["execution"].update(dispatcher_pid=os.getpid(), actual_commit=commit, launch_owner=spec["launch_owner"], runtime_state_ref=str(runtime / "state.json"))
            obj["execution"]["launch_command"] = [spec["python"], "-u", "-m", "comparison_suite.launch", "--spec", str(spec_path), "--commit", commit]
            if commit:
                obj["code"]["commit"] = commit
            lookup = {r["row_id"]: states[row_key(r)] for r in run_rows}
            for row in obj["rows"]:
                if row["row_id"] in lookup:
                    row["runtime"] = lookup[row["row_id"]]
                    row["resolved_config_ref"] = row["output_root"] + "/resolved_config.json"
            atomic_json(record, obj)
            if run_id in original_reports:
                counters = {s: active_statuses.count(s) for s in sorted(set(active_statuses))}
                text = original_reports[run_id] + "\n## Actual dispatcher state\n\n" + \
                    f"Status: `{status}`; owner: `{spec['launch_owner']}`; dispatcher PID: `{os.getpid()}`; release commit: `{commit}`.\n\n" + \
                    f"Counts: `{json.dumps(counters)}`. Per-row PID/GPU/log/config evidence: `{runtime / 'state.json'}` and experiment.json row runtime fields. Scoring: `{json.dumps(score_states)}`.\n"
                (folder / "report.md").write_text(text, encoding="utf-8")
    try:
        occupancy = {} if spec.get("cpu_test_only") else gpu_occupancy()
        event("RUNNING", note="Dispatcher lock held; source/Phase12 lane budgets checked against live GPU compute PIDs", gpu_processes={g:list(p) for g,p in occupancy.items()}, commit=commit)
        if not (runtime / "launch.json").exists():
            atomic_json(runtime / "launch.json", {"pid": os.getpid(), "argv": sys.argv,
                "cwd": str(Path.cwd()), "commit": commit, "launch_owner": spec["launch_owner"]})
        save()
        while True:
            for key, (row, process, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[key]
                ok = code == 0 and completed(row)
                states[key].update(status="TRAINING_COMPLETE" if ok and row["stage"] == "source" else "PREDICTIONS_COMPLETE" if ok else "FAILED", exit_code=code)
                event(states[key]["status"], row, exit_code=code,
                    note="Completion status and required artifact independently read" if ok else "Process ended without verified completion; preserve partial output and do not retry")
            source_by_completion = {str((Path(r["output_root"]) / "completion.json").resolve()): states[row_key(r)] for r in rows if r["stage"] == "source"}
            for row in rows:
                key = row_key(row)
                if states[key]["status"] == "QUEUED" and row["dependency"] and source_by_completion.get(str(Path(row["dependency"]).resolve()), {}).get("status") == "FAILED":
                    states[key]["status"] = "FAILED"
                    event("FAILED", row, note="Source dependency failed; Phase12 was not launched")
            occupancy = {} if spec.get("cpu_test_only") else gpu_occupancy()
            for row in rows:
                if not eligible(row, states, active, occupancy):
                    continue
                Path(row["output_root"]).mkdir(parents=True, exist_ok=True)
                log = Path(row["log"])
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("x", encoding="utf-8")
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(row["gpu"]), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                command = [spec["python"], "-u", "-m", row["module"], "--config", row["config"]]
                process = subprocess.Popen(command, cwd=spec["code_root"], env=env, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                key = row_key(row)
                active[key] = (row, process, handle)
                states[key].update(status="RUNNING", pid=process.pid, command=command, started_at=datetime.now(timezone.utc).isoformat())
                event("RUNNING", row, pid=process.pid, gpu=row["gpu"], command=command,
                    resolved_config_ref=row["output_root"] + "/resolved_config.json", note="Process launched once; raw source/target scores are never inspected by dispatcher")
            pipeline_rows = [r for r in rows if r["stage"] == "phase12"]
            for stage in ("p1", "p2"):
                if stage in scorers:
                    process, handle = scorers[stage]
                    code = process.poll()
                    if code is not None:
                        handle.close()
                        flag = runtime / ("scoring_" + stage + "_complete.json")
                        verified = code == 0 and flag.exists() and read_json(flag).get("status") == "SCORED_COMPLETE"
                        score_states[stage] = "ANALYZED" if verified else "FAILED"
                        event(score_states[stage], note="Independent scoring completion readback", scoring_stage=stage, exit_code=code, artifact=str(flag))
                        del scorers[stage]
                if score_states[stage] == "WAITING" and pipeline_rows and all(completed(r, stage) for r in pipeline_rows):
                    handle = (runtime / ("scoring_" + stage + ".log")).open("x", encoding="utf-8")
                    command = [spec["python"], "-u", "-m", "comparison_suite.score", "--spec", str(spec_path), "--stage", stage]
                    process = subprocess.Popen(command, cwd=spec["code_root"], stdout=handle, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    scorers[stage] = (process, handle)
                    score_states[stage] = "RUNNING"
                    event("RUNNING", note="All stage prediction flags/artifacts verified before starting independent scorer", scoring_stage=stage, pid=process.pid, command=command)
            save()
            if not active and not scorers and all(s["status"] != "QUEUED" for s in states.values()):
                break
            time.sleep(float(spec.get("poll_seconds", 2.)))
        event("ARTIFACTS_COMPLETE" if all(s["status"] != "FAILED" for s in states.values()) and all(s != "FAILED" for s in score_states.values()) else "PARTIAL",
            note="Dispatcher finished; no retry or selective performance resubmission", scoring=score_states)
        save()
    finally:
        lock.unlink(missing_ok=True)
    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        detach(args.spec, args.commit)
    else:
        launch(args.spec, args.commit)
