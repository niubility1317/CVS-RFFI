"""Send a commit-bound release to N607 and start the ordinary-account dispatch."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from .register import COMMAND, CONTEXT, PROJECT, PYTHON, RELEASE, REMOTE
from .plan import RUN
from experiments.cric_dura_comparison.io import write

INSTALL = """
import hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path

request = json.loads(sys.stdin.readline())
rt = request["runtime"]
root, release = Path(rt["run_root"]), Path(rt["release_root"])
archive = Path(request["archive"])
if os.geteuid() == 0 or root.exists() or release.exists():
    raise RuntimeError("Use the ordinary account and an unused release/run root")
if hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() != request["sha256"]:
    raise RuntimeError("Release archive mismatch")
release.mkdir()
with tarfile.open(archive) as package:
    for member in package:
        destination = (release / member.name).resolve()
        if release not in destination.parents or not (member.isfile() or member.isdir()):
            raise ValueError("Unsafe release member")
    package.extractall(release)
ctx = json.loads(Path(rt["context"]).read_text())
rt["frozen_sources"] = ctx["runtime"]["frozen_sources"]
rt_file = Path(rt["runtime_file"])
rt_file.write_text(json.dumps(rt, indent=2) + chr(10), encoding="utf-8")
env = dict(os.environ, PYTHONPATH=str(release), CUDA_VISIBLE_DEVICES="",
           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
subprocess.run([rt["python"], "-m", "compileall", "-q",
                "experiments/pamf_advisor_supplements"], cwd=release, env=env, check=True)
with (release / "controller.log").open("x") as log:
    p = subprocess.Popen(request["command"], cwd=release, env=env, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=p.pid, archive_verified=True, compile_pass=True, cwd=str(release))))
"""


def publish(ssh_config, output, restoration=False):
    import shlex
    if restoration:
        from .dura import RUN as run_id
        release = f"{PROJECT}/releases/{run_id}-release"
        run_root = f"{PROJECT}/runs/{run_id}"
        command = [PYTHON, "-u", "-m", "experiments.pamf_advisor_supplements.dura_dispatch",
                   "dispatch", "--config", f"{release}/runtime.json"]
    else:
        run_id, release, run_root, command = RUN, RELEASE, REMOTE, COMMAND.split()
    root = Path(__file__).resolve().parents[2]
    git = lambda *args: subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("ls-remote", "origin", "refs/heads/" + branch).split()
    if not remote or remote[0] != commit:
        raise ValueError("Release commit unavailable at its Git upstream")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / f"{run_id}-release.tar.gz"
    paths = ["experiments", "comparison_suite", "baselines", "code",
             f"automation_reports/CV-SincNet/{run_id}/experiment.json"]
    with archive.open("xb") as target, gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as zipped:
        from shutil import copyfileobj
        with subprocess.Popen(["git", "-C", str(root), "archive", "--format=tar", commit, "--", *paths],
                              stdout=subprocess.PIPE) as process:
            copyfileobj(process.stdout, zipped)
            if process.wait():
                raise RuntimeError("git archive failed")
    # The run registration lives under its relative repository path.
    rt = dict(run_id=run_id, code_commit=commit, python=PYTHON, project=PROJECT, release_root=release,
              run_root=run_root, runtime_file=f"{release}/runtime.json", context=CONTEXT,
              chrono_cache=f"{PROJECT}/runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/chrono",
              allowed_gpus=[5, 6, 7] if restoration else [0, 1, 2, 4], launch_owner="codex-pamf-advisor-20261124")
    with archive.open("rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    request = dict(runtime=rt, sha256=digest, archive=f"{PROJECT}/releases/{archive.name}",
                   command=command)
    write(output / "request.json", request)
    auth = ["-F", ssh_config, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    subprocess.run(["ssh", *auth, "-T", "N607", "test ! -e " + shlex.quote(request["archive"])], check=True, timeout=20)
    subprocess.run(["scp", *auth, str(archive), f"N607:{request['archive']}"], check=True, timeout=120)
    # The program is fixed; dynamic inputs are transferred through stdin.
    result = subprocess.check_output(["ssh", *auth, "-T", "N607",
        "python3 -c " + shlex.quote(INSTALL)], input=json.dumps(request) + "\n", text=True, timeout=120)
    submit = json.loads(result)
    write(output / "submit.json", submit)
    print(json.dumps(submit))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ssh-config", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--restoration", action="store_true")
    args = p.parse_args()
    publish(args.ssh_config, args.output, args.restoration)
