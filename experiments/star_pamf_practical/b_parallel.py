"""Raise concurrency for the existing B run without restarting model workers."""
import argparse
import os
from pathlib import Path
import signal
import time

from experiments.cric_dura_comparison.io import read, write
from .b_matrix import COMMAND, GPUS, PROJECT, PYTHON, RELEASE, ROOT, RUN, dispatch

PARALLEL = RUN + "-parallel"
CONTROL_RELEASE = PROJECT / "releases" / (PARALLEL + "-release")
MODULE = "experiments.star_pamf_practical.b_parallel"
COMPLETE = {"train": "SOURCE_TRAINED_FROZEN", "predict": "ALL_PREDICTIONS_COMPLETE", "score": "SCORED_COMPLETE"}
GPU_QUERY = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]


def process(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return fields[0], int(fields[1]), fields[19]


def argv(path):
    return (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")


class Adopted:
    def __init__(self, pid, started, path, status):
        self.pid = pid
        self.started = started
        self.path = path
        self.status = status

    def poll(self):
        current = process(self.pid)
        if current and current[2] == self.started and current[0] != "Z":
            return None
        return 0 if self.path.is_file() and read(self.path)["status"] == self.status else 1


def collect(root, state, previous_pid):
    known = state["active"]
    stages = state["stages"]
    parent = Path("/proc") / str(previous_pid)
    children = {int(p) for p in (parent / "task" / str(previous_pid) / "children").read_text().split()}
    for pid in children - {v["pid"] for v in known.values()}:
        # The old queue can be paused while its read-only GPU query is in flight.
        live = process(pid)
        if live and live[0] != "Z" and (live[1] != previous_pid or argv(Path("/proc") / str(pid)) != GPU_QUERY):
            raise RuntimeError("Original dispatcher has an unrecorded child")
    for row, stage in stages.items():
        if row not in known and stage in COMPLETE and (root / "logs" / f"{stage}__{row}.log").exists():
            raise RuntimeError("Original dispatcher has an unrecorded stage log")
    active = {}
    for row, info in known.items():
        stage, pid = info["stage"], info["pid"]
        config = root / "configs" / f"{stage}__{row}.json"
        if stages[row] != stage or info["argv"][-1] != str(config):
            raise ValueError("Original worker does not match its recorded stage")
        live = process(pid)
        child = Path("/proc") / str(pid)
        if live and live[0] != "Z":
            if live[1] != previous_pid or argv(child) != info["argv"] or (child / "cwd").resolve() != RELEASE:
                raise ValueError("Original worker identity or parent differs")
        c = read(config)
        path = Path(c["source_output"] if stage == "train" else c["output_root"]) / "complete.json"
        active[row] = Adopted(pid, live[2] if live else None, path, COMPLETE[stage]), info
    return active


def check_term_default(parent):
    bit = 1 << (signal.SIGTERM - 1)
    masks = {k: int(v, 16) for k, v in (line.split(":", 1) for line in (parent / "status").read_text().splitlines())
             if k in ("SigCgt", "SigIgn", "SigBlk")}
    if set(masks) != {"SigCgt", "SigIgn", "SigBlk"} or any(value & bit for value in masks.values()):
        raise RuntimeError("Original dispatcher does not have the default SIGTERM action")


def wait_for(predicate, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for the original dispatcher")


def handoff(expected_pid, scheduler_commit):
    root = Path(ROOT)
    state_file = root / "pipeline_state.json"
    parent = Path("/proc") / str(expected_pid)
    first = process(expected_pid)
    if (os.geteuid() == 0 or first is None or first[0] in ("T", "t", "Z")
            or parent.stat().st_uid != os.geteuid() or argv(parent) != COMMAND or (parent / "cwd").resolve() != RELEASE):
        raise RuntimeError("Original dispatcher identity is not the authorized target")
    started = first[2]

    def is_original(*states):
        current = process(expected_pid)
        return bool(current and current[2] == started and current[0] in states)

    def between_launches():
        return is_original("S") and (parent / "wchan").read_text().strip() in {
            "hrtimer_nanosleep", "do_nanosleep", "do_select", "pipe_read",
        }

    check_term_default(parent)
    wait_for(between_launches, 30)
    os.kill(expected_pid, signal.SIGSTOP)
    term_sent = False
    try:
        wait_for(lambda: is_original("T"), 5)
        state = read(state_file)
        before = read(RELEASE / "runtime.json")
        if (state["status"] != "RUNNING" or state["controller_pid"] != expected_pid
                or state["code_commit"] != before["code_commit"]):
            raise RuntimeError("Original scheduler state is not the registered running state")
        check_term_default(parent)
        adopted = collect(root, state, expected_pid)
        write(Path(CONTROL_RELEASE) / "handoff_before.json", state)
        # Default, unblocked SIGTERM is pending before the old dispatcher continues.
        # Only the dispatcher PID is signaled. Its workers keep running.
        os.kill(expected_pid, signal.SIGTERM)
        term_sent = True
        os.kill(expected_pid, signal.SIGCONT)
        wait_for(lambda: process(expected_pid) is None or is_original("Z", "X"), 10)
    finally:
        if not term_sent and is_original("T"):
            os.kill(expected_pid, signal.SIGCONT)
    write(Path(CONTROL_RELEASE) / "handoff_complete.json", dict(status="ACTIVE", previous_pid=expected_pid,
          controller_pid=os.getpid(), adopted={k: p.pid for k, (p, _) in adopted.items()},
          original_commit=before["code_commit"], scheduler_commit=scheduler_commit, gpu_workers=len(GPUS)))
    dispatch(dict(before, scheduler_commit=scheduler_commit), resume_state=state,
             adopted=adopted, max_workers=len(GPUS))


def publish(config, output, expected_pid):
    from .publish import publish_bundle

    command = [PYTHON, "-u", "-m", MODULE, "handoff", "--expected-pid", str(expected_pid),
               "--config", str(CONTROL_RELEASE / "runtime.json")]
    paths = ["experiments/star_pamf_practical", "experiments/pamf_comparison_matrix",
             "experiments/pamf_advisor_supplements/plan.py", "experiments/practical_full_2x2",
             "experiments/cric_dura_comparison", "code/leo_practical"]
    return publish_bundle(config, output, run=PARALLEL, project=PROJECT, release=CONTROL_RELEASE,
                          run_root=ROOT.parent / PARALLEL, python=PYTHON, command=command, paths=paths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    actions = p.add_subparsers(dest="action", required=True)
    switch = actions.add_parser("handoff")
    switch.add_argument("--expected-pid", type=int, required=True)
    switch.add_argument("--config", type=Path, required=True)
    release = actions.add_parser("publish")
    release.add_argument("--expected-pid", type=int, required=True)
    release.add_argument("--ssh-config", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "publish":
        publish(a.ssh_config, a.output, a.expected_pid)
    else:
        handoff(a.expected_pid, read(a.config)["code_commit"])
