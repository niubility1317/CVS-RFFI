"""Expand the advisor queue's GPUs while retaining its detached workers."""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import sys
import time

WORKER = "experiments.pamf_advisor_supplements.runner"
ALL_GPUS = list(range(8))


def args_of(proc):
    return [value.decode() for value in (proc / "cmdline").read_bytes().split(b"\0") if value]


def process_state(proc):
    # The command name in /proc/PID/stat is bracketed and can contain spaces.
    return (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]


def alive(pid):
    try:
        return process_state(Path("/proc") / str(pid)) != "Z"
    except FileNotFoundError:
        return False


def reconcile(rt, previous, jobs, done):
    state = {key: dict(item) for key, item in previous["jobs"].items()}
    by_config = {j["config"]: j for j in jobs}
    active = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = args_of(proc)
            if len(argv) != 9 or argv[1:5] != ["-u", "-m", WORKER, "worker"]:
                continue
            job = by_config.get(argv[-1])
            if job is None:
                continue
            expected = [rt["python"], "-u", "-m", WORKER, "worker",
                        "--kind", job["kind"], "--config", job["config"]]
            if argv != expected or os.readlink(proc / "cwd") != rt["release_root"]:
                raise ValueError(f"Existing worker identity differs for {job['id']}")
            env = dict(v.split(b"=", 1) for v in (proc / "environ").read_bytes().split(b"\0") if b"=" in v)
            visible = env[b"CUDA_VISIBLE_DEVICES"].decode()
            gpu = None if job["kind"] == "score" else int(visible)
            if gpu not in (*rt["allowed_gpus"], None) or (gpu is None and visible):
                raise ValueError(f"Existing worker GPU differs for {job['id']}")
            key = job["id"]
            old = state[key]
            if key in active or (old.get("pid") not in (None, int(proc.name))):
                raise ValueError(f"Conflicting live worker for {key}")
            active[key] = dict(old, status="RUNNING", pid=int(proc.name), gpu=gpu,
                               argv=argv, cwd=rt["release_root"], adopted=True,
                               log=str(Path(rt["run_root"]) / "logs" / f"{key}.log"))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    root = Path(rt["run_root"])
    for job in jobs:
        key = job["id"]
        if key in active:
            state[key] = active[key]
        elif done(job):
            state[key].update(status="COMPLETE")
        elif state[key]["status"] != "QUEUED" or (root / "logs" / f"{key}.log").exists():
            raise RuntimeError(f"Existing job without process or completion: {key}")
    slots = [item["gpu"] for item in active.values()]
    if len(slots) != len(set(slots)):
        raise ValueError("More than one existing worker in a dispatch slot")
    return state, active


def take_over(rt, path, jobs, done, read):
    prior = read(path)
    pid = prior["pid"]
    proc = Path("/proc") / str(pid)
    expected = [rt["python"], "-u", "-m", WORKER, "dispatch", "--config", rt["runtime_file"]]
    if (prior["status"] != "RUNNING" or prior.get("failed_jobs") or prior["code_commit"] != rt["code_commit"]
            or set(prior["jobs"]) != {j["id"] for j in jobs}
            or proc.stat().st_uid != os.getuid() or args_of(proc) != expected
            or os.readlink(proc / "cwd") != rt["release_root"]):
        raise ValueError("Ownership or current queue state differs")
    # Pause the dispatcher only. Its workers have their own sessions.
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(30):
            if process_state(proc) == "T":
                break
            time.sleep(.1)
        else:
            raise RuntimeError("Failed to pause the original dispatcher")
        latest = read(path)
        if latest["pid"] != pid or latest.get("failed_jobs"):
            raise RuntimeError("Original queue changed during handover")
        state, active = reconcile(rt, latest, jobs, done)
        if not active:
            raise RuntimeError("No active workers to adopt")
        os.kill(pid, signal.SIGTERM)
    finally:
        os.kill(pid, signal.SIGCONT)
    for _ in range(50):
        if not alive(pid):
            return dict(latest, jobs=state), active
        time.sleep(.1)
    raise RuntimeError("Previous dispatcher remains alive")


def run(rt, commit):
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.pipeline import block_failed_dependencies, done, ready
    from experiments.pamf_advisor_supplements.runner import admits, analyze, launch
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    from experiments.practical_full_2x2.cache import atomic

    root = Path(rt["run_root"])
    path = root / "pipeline_state.json"
    jobs = read(root / "jobs.json")
    by_id = {job["id"]: job for job in jobs}
    if read(root / "smoke.json")["status"] != "PASS":
        raise ValueError("Original immutable worker release lacks a passing smoke")
    if not any(admits(gpu) for gpu in ALL_GPUS if gpu not in rt["allowed_gpus"]):
        raise ValueError("No additional card currently has enough memory")
    state, active = take_over(rt, path, jobs, done, read)
    state.update(previous_dispatcher=state["pid"], pid=os.getpid(), control_commit=commit,
                 allowed_gpus=ALL_GPUS, adopted_workers=sorted(active), cwd=os.getcwd(), argv=sys.argv)
    atomic(path, state)
    children = {}
    failures = []
    while True:
        for key, item in list(active.items()):
            child = children.get(key)
            running = child.poll() is None if child is not None else alive(item["pid"])
            if running:
                continue
            ok = done(by_id[key])
            state["jobs"][key] = dict(item, status="COMPLETE" if ok else "FAILED")
            if not ok:
                failures.append(key)
            del active[key]
            children.pop(key, None)
        block_failed_dependencies(jobs, state, [])
        queued = [j for j in jobs if state["jobs"][j["id"]]["status"] == "QUEUED"]
        eligible = [j for j in queued if ready(j)] if not failures else []
        score = next((j for j in eligible if j["kind"] == "score"), None)
        assignments = []
        if score is not None and not any(item["gpu"] is None for item in active.values()):
            if not foreign_CPU_score({item["pid"] for item in active.values()}):
                assignments.append((score, None))
        gpu_jobs = sorted((j for j in eligible if j["kind"] != "score"),
                          key=lambda j: j["kind"] != "predict")
        for gpu in ALL_GPUS:
            if not gpu_jobs:
                break
            if not any(item["gpu"] == gpu for item in active.values()) and admits(gpu):
                assignments.append((gpu_jobs.pop(0), gpu))
        for job, gpu in assignments:
            key = job["id"]
            try:
                child, item = launch(rt, job, gpu)
            except (OSError, ValueError) as error:
                state["jobs"][key] = dict(status="FAILED", error=str(error))
                failures.append(key)
                break
            children[key], active[key], state["jobs"][key] = child, item, item
        state.update(updated_at=time.time(), counts=dict(Counter(v["status"] for v in state["jobs"].values())),
                     GPU_workers=sum(item["gpu"] is not None for item in active.values()),
                     failed_jobs=failures, stop_new_dispatch=bool(failures))
        atomic(path, state)
        if not active and (not queued or failures):
            break
        time.sleep(15)
    state["status"] = "PARTIAL_TECHNICAL_FAILURE" if failures else "ANALYZING"
    atomic(path, state)
    if not failures:
        analyze(rt)
        state["status"] = "ANALYZED"
        atomic(path, state)


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    sys.path.insert(0, runtime["release_root"])
    run(runtime, args.commit)
