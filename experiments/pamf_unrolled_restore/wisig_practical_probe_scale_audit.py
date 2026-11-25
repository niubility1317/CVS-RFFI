"""Rescale sparse fitting with archived, received-only Practical source U views."""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_sparse_audit as sparse
from . import wisig_lora_source_fit as legacy_source


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r06"
SEED = 2026112606
FIT, HOLD = 768, 192
COUNT = 63000
matched = sparse.matched


def earliest_views(scenes):
    for index, names in enumerate(scenes):
        if index == 0:
            epochs = np.zeros((len(matched.SCENES), len(names)), dtype=np.int8)
        for slot, name in enumerate(matched.SCENES):
            take = (epochs[slot] == 0) & (names == name)
            epochs[slot, take] = index + 1
    return epochs


def selection(epochs, roles, ids, excluded):
    eligible = np.flatnonzero((roles == "U_s") & np.all(epochs > 0, axis=0) & ~np.isin(ids, excluded))
    if len(eligible) < FIT + HOLD:
        raise ValueError("Not enough independent source-U packets in all six Practical scenes")
    return np.random.default_rng(SEED).choice(eligible, FIT + HOLD, replace=False)


def prepare_action_views(cache, legacy, root):
    contract = read(cache / "complete.json")
    if (contract["status"], contract["train_count"], contract["epochs"], contract["target_reads"]) != (
            "ACTION_SOURCE_CACHE_COMPLETE", COUNT, 20, 0):
        raise ValueError("Expected the fixed R07 source archive")
    roles, ids = (np.load(cache / "train" / f"{key}.npy", allow_pickle=False) for key in ("roles", "physical_ids"))
    old_roles, old_ids = (np.load(legacy / "train" / f"{key}.npy", allow_pickle=False) for key in ("roles", "physical_ids"))
    if any(a.shape != (COUNT,) for a in (roles, ids, old_roles, old_ids)) or len(np.unique(ids)) != COUNT:
        raise ValueError("Unexpected R07 source metadata layout")
    base = np.random.default_rng(legacy_source.SEED).permutation(np.flatnonzero(old_roles == "U_s"))
    old_n = matched.protocol.panel.NDEV
    old_seed = matched.protocol.panel.SEED
    legacy_indices = np.random.default_rng(old_seed).choice(base[legacy_source.CALIBRATION:], old_n, replace=False)

    def archive_scenes():
        for epoch in range(1, contract["epochs"] + 1):
            with np.load(cache / "epochs" / f"{epoch:02d}" / "groups.npz", allow_pickle=False) as group:
                names = group["scene"]
                if names.shape != (COUNT,):
                    raise ValueError("Unexpected Practical scene index")
                yield names

    epochs = earliest_views(archive_scenes())
    selected = selection(epochs, roles, ids, old_ids[legacy_indices])
    by_scene = epochs[:, selected]
    output = np.empty((len(matched.SCENES), len(selected), matched.model.receive.SAMPLES), dtype=np.complex64)
    for epoch in np.unique(by_scene):
        received = np.load(cache / "epochs" / f"{int(epoch):02d}" / "received.npy", mmap_mode="r", allow_pickle=False)
        if received.shape != (COUNT, 2, output.shape[-1]) or received.dtype != np.float32:
            raise ValueError("Unexpected cache of received R07 IQ")
        for slot, view in enumerate(by_scene):
            chosen = view == epoch
            data = received[selected[chosen]]
            output[slot, chosen] = data[:, 0] + 1j * data[:, 1]
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite archived Practical received IQ")
    np.save(root / "source_received.npy", output, allow_pickle=False)
    write(root / "source.json", dict(physical_fit=FIT, physical_hold=HOLD, scenes=list(matched.SCENES),
          distinct_physical=len(np.unique(ids[selected])), legacy_physical_overlap=0,
          allowed_iq="only Practical received", preparation_metadata=["source role", "physical ID", "epoch scene"]))
    print("R07_PRACTICAL_SOURCE_SCALE_READY", len(selected), len(matched.SCENES), flush=True)


def tune(source, first, second, third, fourth, output):
    bank = np.load(source, allow_pickle=False)
    if bank.shape != (len(matched.SCENES), FIT + HOLD, matched.model.receive.SAMPLES):
        raise ValueError("Expected the scaled receive-only Practical source bank")
    fit, hold = np.split(np.arange(FIT + HOLD), [FIT])
    sparse.fit_bank(bank, fit, hold, first, second, third, fourth, output)


def dispatch(runtime):
    rt = read(runtime)
    root, first, second, third, fourth, cache, legacy = (Path(rt[k]) for k in (
        "run_root", "first", "second", "third", "fourth", "cache", "legacy"))
    p1, p2, p3 = (p / "configuration.npz" for p in (first, second, third))
    opts = root / "configuration.npz"
    steps = (("smoke", ()), ("prepare_action_views", (cache, legacy, root)),
             ("tune", (root / "source_received.npy", p1, p2, p3, fourth / "configuration.json", opts)),
             ("predict", (first / "evaluation_received.npy", p1, p3, opts, root / "estimates.npz")),
             ("score", (root, cache / "validation_actions")))
    matched.protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    first = sparse.previous
    inputs = dict(first=str(runs / matched.RUN), second=str(runs / first.gate_audit.relative.RUN),
                  third=str(runs / first.gate_audit.RUN), fourth=str(runs / first.RUN),
                  cache=str(runs / matched.PARENT / "cache"),
                  legacy=str(runs / "20261019-phase1-reception-restoration-selfsup-manysig-m6-r02" / "cache"))
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN, runner=__spec__.name,
              code_commit=transport.release_commit(repo),
              release_root=str(runs.parent / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              external_releases=list(inputs.values()), **inputs)
    transport.release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="action", required=True)
    pub = sub.add_parser("publish")
    for name in ("repo", "output", "context", "ssh-config"):
        pub.add_argument("--" + name, type=Path, required=True)
    steps = ((prepare_action_views, ("cache", "legacy", "root")),
             (tune, ("source", "first", "second", "third", "fourth", "output")),
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
