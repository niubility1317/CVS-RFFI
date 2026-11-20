"""Transfer the pending SatIQ E100 dispatcher to physical GPUs 2 and 3."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess


OLD_PID = 2527656
TRAINING_OWNER = 2248959
TRAINERS = {"SATIQ": 2249306, "PAMF_SUP": 2249305}
RELEASE = Path("/home/szu2070436088/2510044040/CV-SincNet/releases/20261120-phase1-satiq-matched-single-packet-e100-s2026092701-r01")
OLD_COMMAND = ["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-u","-m","experiments.pamf_satiq66.paper_single_packet_dispatch","/home/szu2070436088/2510044040/CV-SincNet/data/satiq_oxford_ccs2023_supervised_20261119_r01","/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-ccs2023-header-matched-sup200-s2026092701-r01","/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-matched-single-packet-e100-s2026092701-r01","--owner","2248959","--gpus","2","3","4","1","5","6","7","0","--epoch","100"]
NEW_COMMAND = ["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-u","-m","experiments.pamf_satiq66.paper_single_packet_dispatch","/home/szu2070436088/2510044040/CV-SincNet/data/satiq_oxford_ccs2023_supervised_20261119_r01","/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-ccs2023-header-matched-sup200-s2026092701-r01","/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-matched-single-packet-e100-s2026092701-r02","--owner","2248959","--gpus","2","3","--epoch","100"]
TRAINING = Path(NEW_COMMAND[5])
OLD_OUTPUT = Path(OLD_COMMAND[6])
NEW_OUTPUT = Path(NEW_COMMAND[6])
LOG_ROOT = Path("/home/szu2070436088/2510044040/CV-SincNet/releases/20261120-phase1-satiq-matched-single-packet-e100-s2026092701-r02")
CHECKPOINT = "s2822500.pt"


def command(pid):
    return (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")


def last_updates(steps):
    with steps.open("rb") as log:
        end = log.seek(0, os.SEEK_END)
        log.seek(max(0, end - 65536))
        recent = log.read()
    complete = recent[:recent.rfind(b"\n")]
    return json.loads(complete.rsplit(b"\n", 1)[-1])["updates"]


def preflight():
    dispatcher = Path("/proc") / str(OLD_PID)
    if command(OLD_PID) != OLD_COMMAND or (dispatcher / "cwd").resolve(strict=True) != RELEASE:
        raise RuntimeError("Old waiting process identity changed")
    if (dispatcher / "task" / str(OLD_PID) / "children").read_text().strip() or any(OLD_OUTPUT.iterdir()):
        raise RuntimeError("Old dispatcher started handling an E100 milestone")
    if NEW_OUTPUT.exists() or LOG_ROOT.exists():
        raise FileExistsError("Replacement output or log path exists")
    owner = json.loads((TRAINING / "owner.json").read_text())
    owner_command = command(TRAINING_OWNER)
    if owner["pid"] != TRAINING_OWNER or json.loads(owner_command[-1])["root"] != str(TRAINING):
        raise RuntimeError("Original training owner changed")
    roles = {row["name"]: row for row in owner["roles"]}
    updates = {}
    for arm, pid in TRAINERS.items():
        row = TRAINING / arm
        trainer = Path("/proc") / str(pid)
        status = (trainer / "status").read_text().splitlines()
        parent = next(line.split()[1] for line in status if line.startswith("PPid:"))
        if (int(parent) != TRAINING_OWNER or int((TRAINING / f"{arm}.pid").read_text()) != pid
                or command(pid) != roles[arm]["cmd"].split()):
            raise RuntimeError(f"{arm} trainer identity changed")
        if (row / CHECKPOINT).exists() or (TRAINING / f"{arm}.exit").exists():
            raise RuntimeError(f"{arm} already reached the handoff boundary")
        updates[arm] = last_updates(row / "steps.jsonl")
        if updates[arm] >= 99 * 28225:
            raise RuntimeError(f"{arm} is too close to E100 for the waiting-only handoff")
    return updates


def apply():
    pidfd = os.pidfd_open(OLD_PID)
    try:
        updates = preflight()
        LOG_ROOT.mkdir(exist_ok=False)
        with (LOG_ROOT / "dispatch.log").open("xb") as log:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            if not select.select([pidfd], [], [], 10)[0]:
                raise TimeoutError("Old dispatcher has not exited")
            worker = subprocess.Popen(
                NEW_COMMAND, cwd=RELEASE, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
        (LOG_ROOT / "dispatch.pid").write_text(f"{worker.pid}\n", encoding="utf-8")
    finally:
        os.close(pidfd)
    return {"old_dispatcher_exited": OLD_PID, "new_dispatcher": worker.pid, "trainer_updates": updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        result = apply()
    else:
        result = {"trainer_updates": preflight(), "output_free_bytes": shutil.disk_usage(NEW_OUTPUT.parent).free}
    print(json.dumps({"at": datetime.now().astimezone().isoformat(timespec="seconds"), **result}))


if __name__ == "__main__":
    main()
