"""Score jointly reconstructed source anchors on the fixed R07 comparison."""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_joint as joint
from . import wisig_practical_probe_scale_audit as scale


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r08"
matched = scale.matched
sparse = scale.sparse


def tune(source, first, second, third, fourth, output):
    bank = np.load(source, allow_pickle=False)
    if bank.shape != (len(matched.SCENES), scale.EXPANDED_FIT + scale.HOLD, matched.model.receive.SAMPLES):
        raise ValueError("Expected the frozen 4096/192 received source bank")
    fit, held = np.split(np.arange(bank.shape[1]), [scale.EXPANDED_FIT])
    sparse.fit_bank(bank, fit, held, first, second, third, fourth, output,
                    ridge_options=(*sparse.RIDGES, 1600., 3200.), compact_builder=joint.target_columns)


def dispatch(runtime):
    rt = read(runtime)
    root, first, second, third, fourth, source, reference = [Path(rt[k]) for k in (
        "run_root", "first", "second", "third", "fourth", "source", "reference")]
    p1, p2, p3 = (p / "configuration.npz" for p in (first, second, third))
    opts = root / "configuration.npz"
    steps = (("smoke", ()), ("tune", (source, p1, p2, p3, fourth / "configuration.json", opts)),
             ("predict", (first / "evaluation_received.npy", p1, p3, opts, root / "estimates.npz")),
             ("score", (root, reference)))
    matched.protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    previous = sparse.previous
    paths = dict(first=str(runs / matched.RUN), second=str(runs / previous.gate_audit.relative.RUN),
                 third=str(runs / previous.gate_audit.RUN), fourth=str(runs / previous.RUN),
                 source=str(runs / scale.EXPANDED_RUN / "source_received.npy"),
                 reference=str(runs / matched.PARENT / "cache/validation_actions"))
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN, runner=__spec__.name,
              code_commit=transport.release_commit(repo),
              release_root=str(runs.parent / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              external_releases=list(paths.values()), **paths)
    transport.release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="action", required=True)
    pub = sub.add_parser("publish")
    for name in ("repo", "output", "context", "ssh-config"):
        pub.add_argument("--" + name, type=Path, required=True)
    steps = ((tune, ("source", "first", "second", "third", "fourth", "output")),
             (sparse.predict, ("received", "first", "third", "params", "output")),
             (sparse.score, ("root", "reference")), (dispatch, ("runtime",)))
    sub.add_parser("smoke")
    for f, fields in steps:
        stage = sub.add_parser(f.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    command = args.pop("action")
    {"publish": publish, "smoke": matched.smoke, **{f.__name__: f for f, _ in steps}}[command](**args)


if __name__ == "__main__":
    main()
