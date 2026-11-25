"""Confirm one frozen sparse estimator across every WiSig channel family."""

import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main import publish as transport
from . import wisig_joint_sparse_audit as joint
from . import wisig_path_motion_audit as source


RUN = "20261125-diagnostic-wisig-joint-sparse-full-s2026112511-r01"
SEED = 2026112511
WORKERS = 8
THRESHOLD = .5


def prepare(root, cache):
    _, used, _ = source.fresh_source(cache)
    source.prepare(root, cache, SEED, used)


def predict(paired, config, output):
    slots = range(len(source.parent.LABELS))
    joint.predict_matrix(paired, config, THRESHOLD, output, slots, WORKERS)


def score(root):
    source.parent.score(root, joint.METHODS, SEED, "Zero physical ID overlap with both earlier source-V panels")


def smoke(config):
    joint.smoke(config, THRESHOLD)


def dispatch(runtime):
    rt = read(runtime)
    root, config = Path(rt["run_root"]), Path(rt["configuration"])
    stages = (("smoke", (config,)), ("prepare", (root, rt["source_cache"])),
              ("predict", (root / "paired.npz", config, root / "estimates.npz")), ("score", (root,)))
    source.parent.parent.prior_runner.dispatch(rt, stages)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    base = PurePosixPath(ctx["project"])
    runs = base / "runs"
    cache = runs / source.practical.SOURCE / "cache"
    config = runs / source.parent.RUN / "configuration.json"
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN,
              runner=__spec__.name, code_commit=transport.release_commit(repo),
              release_root=str(base / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              source_cache=str(cache), configuration=str(config),
              external_releases=[str(cache), str(config)])
    transport.release(repo, output, rt, ssh_config)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    publish_cmd = commands.add_parser("publish")
    for arg in ("repo", "output", "context", "ssh-config"):
        publish_cmd.add_argument("--" + arg, type=Path, required=True)
    stages = ((smoke, ("config",)), (prepare, ("root", "cache")),
              (predict, ("paired", "config", "output")), (score, ("root",)), (dispatch, ("runtime",)))
    for function, fields in stages:
        command = commands.add_parser(function.__name__)
        for field in fields:
            command.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    {"publish": publish, **{function.__name__: function for function, _ in stages}}[action](**args)


if __name__ == "__main__":
    main()
