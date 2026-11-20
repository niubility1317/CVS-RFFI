"""Use the host Python's Linux pidfd interface to stop one registered trainer."""

import json
import os
from pathlib import Path
import shlex
import signal
import sys


def request_stop(pid, owner, command):
    process = Path("/proc") / str(pid)
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        try:
            argv = process.joinpath("cmdline").read_bytes().strip(b"\0").decode().split("\0")
            status = process.joinpath("status").read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        if argv == [""]:
            return False
        parent = int(next(line for line in status.splitlines() if line.startswith("PPid:")).split()[1])
        if parent != owner or argv != shlex.split(command):
            raise ValueError(f"PID {pid} is not the registered trainer")
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    pid, owner, command = sys.argv[1:]
    print(json.dumps(request_stop(int(pid), int(owner), command)))
