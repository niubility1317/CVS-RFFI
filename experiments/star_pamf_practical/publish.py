"""Transfer one commit-bound archive and start the N607 dispatcher."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from .runner import COMMAND, PROJECT, PYTHON, RELEASE, RUN, RUN_ROOT

INSTALL = """
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tarfile

r = json.loads(sys.stdin.readline())
release, output = Path(r["release"]), Path(r["run_root"])
if os.geteuid() == 0 or release.exists() or output.exists():
    raise RuntimeError("Ordinary account and unused release/run paths required")
with Path(r["archive"]).open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if digest != r["sha256"]:
    raise ValueError("Archive transfer checksum mismatch")
release.mkdir()
with tarfile.open(r["archive"]) as archived:
    for member in archived:
        child = (release / member.name).resolve()
        if release not in child.parents or not (member.isdir() or member.isfile()):
            raise ValueError("Unexpected archive member")
    archived.extractall(release)
runtime = release / "runtime.json"
with runtime.open("x") as f:
    json.dump(dict(code_commit=r["commit"]), f)
env = dict(os.environ, PYTHONPATH=str(release) + os.pathsep + str(release / "code"),
           CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
subprocess.run([r["python"], "-m", "compileall", "-q", "experiments", "code/leo_practical"],
               cwd=release, env=env, check=True)
with (release / "controller.log").open("x") as log:
    proc = subprocess.Popen(r["command"], cwd=release, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(status="SUBMITTED", pid=proc.pid, cwd=str(release),
                      compile_pass=True, archive_verified=True, commit=r["commit"])))
"""


def publish_bundle(config, output, *, run, project, release, run_root, python, command, paths):
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("ls-remote", "origin", f"refs/heads/{branch}").split()
    if not remote or remote[0] != commit:
        raise ValueError("Release commit must be available from the branch upstream")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / f"{run}-{commit[:12]}-release.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", f"--output={archive}", commit, "--", *paths],
                   check=True)
    remote_archive = project / "releases" / archive.name
    auth = ["-F", str(config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    ssh = ["ssh", *auth, "-T", "N607"]
    probe = " && ".join("test ! -e " + shlex.quote(str(path))
                        for path in (remote_archive, release, run_root))
    subprocess.run([*ssh, probe], check=True, timeout=25)
    subprocess.run(["scp", *auth, str(archive), f"N607:{remote_archive}"], check=True, timeout=120)
    request = dict(release=str(release), run_root=str(run_root), archive=str(remote_archive),
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), commit=commit,
                   python=python, command=command)
    (output / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    submitted = subprocess.check_output([*ssh, "python3 -c " + shlex.quote(INSTALL)],
                                        input=json.dumps(request) + "\n", text=True, timeout=120)
    (output / "submit.json").write_text(submitted, encoding="utf-8")
    print(submitted, end="")


def publish(config, output):
    paths = ["experiments/star_pamf_practical", "experiments/practical_full_2x2",
             "experiments/cric_dura_comparison", "code/leo_practical",
             f"automation_reports/CV-SincNet/{RUN}/experiment.json"]
    publish_bundle(config, output, run=RUN, project=PROJECT, release=RELEASE, run_root=RUN_ROOT,
                   python=PYTHON, command=COMMAND, paths=paths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ssh-config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    opts = p.parse_args()
    publish(opts.ssh_config, opts.output)
