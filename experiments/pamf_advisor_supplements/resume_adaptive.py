"""Adopt the already running advisor workers and prioritize queued Adaptive Tri."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

from adaptive_queue import SOURCE_RELEASE, free_memory, schedule
from capacity_handoff import alive, args_of, process_state

MIN_FREE_MIB = 8000


def worker_args(rt, kind, config):
    return [rt["python"], "-u", str(Path(rt["release_root"])/"adaptive_queue.py"), "worker",
            "--runtime", rt["runtime_file"], "--kind", kind, "--config", config]


def adopt(rt, state, jobs, done):
    active = {}
    by_config = {job["config"]:job for job in jobs.values()}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or process_state(proc) == "Z":
                continue
            argv = args_of(proc)
            job = by_config.get(argv[-1]) if argv else None
            if job is None or argv != worker_args(rt, job["kind"], job["config"]):
                continue
            key = job["id"]
            row = state["jobs"][key]
            if (key in active or row.get("pid") not in (None, int(proc.name))
                    or os.readlink(proc/"cwd") != rt["release_root"]):
                raise ValueError(f"Conflicting worker identity for {key}")
            env = dict(item.split(b"=", 1) for item in (proc/"environ").read_bytes().split(b"\0") if b"=" in item)
            device = env[b"CUDA_VISIBLE_DEVICES"].decode()
            gpu = int(device) if device else None
            if (gpu is None) != (job["kind"] in ("score", "reuse")) or gpu not in (None, *rt["allowed_gpus"]):
                raise ValueError(f"Wrong GPU for {key}")
            log = Path(rt["run_root"])/"logs"/f"{key}.log"
            active[key] = None, dict(row, status="RUNNING", pid=int(proc.name), gpu=gpu, argv=argv,
                                     cwd=rt["release_root"], log=str(log), adopted=True)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    slots = [item["gpu"] for _,item in active.values()]
    if len(slots) != len(set(slots)):
        raise ValueError("Duplicate active GPU slot")
    for key, job in jobs.items():
        row = state["jobs"][key]
        if key in active:
            state["jobs"][key] = active[key][1]
        elif done(job):
            row["status"] = "COMPLETE"
        elif row["status"] != "QUEUED" or (Path(rt["run_root"])/"logs"/f"{key}.log").exists():
            raise ValueError(f"No live worker or completion for {key}")
        if job["kind"] == "train" and "-adaptive-" in key:
            job["priority"] = -1
    return active


def run(rt, commit):
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.pipeline import done, ready
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    from experiments.practical_full_2x2.cache import atomic

    root = Path(rt["run_root"])
    path = root/"pipeline_state.json"
    jobs = {job["id"]:job for job in read(root/"jobs.json")}
    state = read(path)
    pid = state["pid"]
    proc = Path("/proc")/str(pid)
    expected = worker_args(rt, "", "")[:3] + ["dispatch", "--runtime", rt["runtime_file"]]
    if (state["status"] != "RUNNING" or state.get("failed_jobs") or state["code_commit"] != rt["code_commit"]
            or set(state["jobs"]) != set(jobs) or read(root/"smoke.json")["status"] != "PASS"
            or proc.stat().st_uid != os.getuid() or args_of(proc) != expected
            or os.readlink(proc/"cwd") != rt["release_root"]):
        raise ValueError("Original immutable queue or smoke does not match")
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(30):
            if process_state(proc) == "T":
                break
            time.sleep(.1)
        else:
            raise RuntimeError("Original controller did not pause")
        state = read(path)
        if state.get("pid") != pid or state["status"] != "RUNNING" or state.get("failed_jobs"):
            raise ValueError("Original controller changed during handoff")
        active = adopt(rt, state, jobs, done)
        busy = {item["gpu"] for _,item in active.values()}
        slots = [gpu for gpu in rt["allowed_gpus"] if gpu not in busy and free_memory(gpu) >= MIN_FREE_MIB]
        waiting = any(job.get("priority") == -1 and state["jobs"][key]["status"] == "QUEUED" and ready(job)
                      for key,job in jobs.items())
        if not slots or not waiting:
            print("CAPACITY_HANDOFF_DEFERRED", flush=True)
            return
        os.kill(pid, signal.SIGTERM)
    finally:
        os.kill(pid, signal.SIGCONT)
    for _ in range(50):
        if not alive(pid):
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Original controller still active")
    state.update(pid=os.getpid(), previous_dispatcher=pid, control_commit=commit,
                 training_min_free_mib=MIN_FREE_MIB, prioritized="adaptive_training", adopted_workers=sorted(active))
    atomic(path, state)
    schedule(rt, jobs, state, active,
             lambda memory, job: memory >= (MIN_FREE_MIB if job["kind"] == "train" else 11000),
             foreign_score=foreign_CPU_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    import json

    config = json.loads(args.runtime.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(config["project"])/"releases"/SOURCE_RELEASE))
    run(config, args.commit)
