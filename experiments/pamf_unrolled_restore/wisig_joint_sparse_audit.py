"""Compare the joint sparse branch with the packet-adaptive motion estimator."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_joint_sparse as solver
from . import wisig_path_motion_audit as source
from .wisig_independent_channel_audit import N, TEST


RUN = "20261125-diagnostic-wisig-joint-sparse-s2026112510-r01"
FAMILIES = ("sparse_doppler", "multipath_burst", "fractional_clock", "time_selective")
LABELS = tuple(label for label in source.parent.LABELS if label.split("/")[0] in FAMILIES)
SLOTS = tuple(source.parent.LABELS.index(label) for label in LABELS)
SHAPE = (len(SLOTS), source.panel.NTEST)
METHODS = ("adaptive_path", "joint_sparse")


def smoke(config, threshold):
    options = read(config)
    rng = np.random.default_rng(819)
    clean = rng.normal(size=N) + 1j * rng.normal(size=N)
    basis = clean[solver.OFFSETS]
    reference = np.exp(.61j * solver.U) * (basis[:, 24] + (.2 + .5j) * basis[:, 36]
                                          + (.13 - .2j) * basis[:, 43])
    received = reference + .02 * (rng.normal(size=N) + 1j * rng.normal(size=N))
    received[5::21] += .7 * (1 + 1j)
    baseline, candidate, variance, weight = solver.estimate(clean, received, options, float(threshold))
    error = np.sum(abs(candidate - reference)**2) / np.sum(abs(reference)**2)
    if not (np.isfinite(baseline).all() and error < .001 and variance > 0 and 0. <= weight <= 1.):
        raise ValueError("Joint sparse IQ-only smoke did not recover the known sparse channel")
    print("JOINT_SPARSE_NO_QUERY_SMOKE_PASS", flush=True)


def predict_packet(job):
    x, y, options, threshold = job
    old, new, variance, weight = solver.estimate(x, y, options, threshold)
    return old[TEST], new[TEST], variance, weight


def predict_matrix(paired, config, threshold, output, slots, workers):
    with np.load(paired, allow_pickle=False) as stored:
        clean, received = stored["clean"], stored["received"]
    if clean.shape != (SHAPE[1], N) or received.shape != (*source.parent.SHAPE, N):
        raise ValueError("Expected the registered source-V IQ matrix")
    options = read(config)
    penalty = float(threshold)
    shape = (len(slots), source.panel.NTEST)
    waves = {name: np.empty((*shape, len(TEST)), complex) for name in METHODS}
    variance, weight = np.empty(shape), np.empty(shape)
    with ProcessPoolExecutor(workers) as pool:
        for index, slot in enumerate(slots):
            jobs = ((x, y, options, penalty) for x, y in zip(clean, received[slot]))
            rows = list(pool.map(predict_packet, jobs, chunksize=3))
            waves["adaptive_path"][index], waves["joint_sparse"][index], variance[index], weight[index] = zip(*rows)
            print("V_ESTIMATES_CLOSED", source.parent.LABELS[slot], len(rows), flush=True)
    np.savez(output, **waves, variance=variance, weight=weight)


def predict(paired, config, threshold, output):
    predict_matrix(paired, config, threshold, output, SLOTS, source.WORKERS)


def score(root, reference, threshold):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Scoring uses a separate CPU process")
    with np.load(root / "estimates.npz", allow_pickle=False) as stored:
        waves = {name: stored[name] for name in METHODS}
        variance, weight = stored["variance"], stored["weight"]
    if (variance.shape != SHAPE or weight.shape != SHAPE or (variance <= 0).any()
            or not np.isfinite(variance).all() or not ((0 <= weight) & (weight <= 1)).all()
            or any(wave.shape != (*SHAPE, len(TEST)) or not np.isfinite(wave).all() for wave in waves.values())):
        raise ValueError("Freeze two complete predictions before reference access")
    with np.load(reference, allow_pickle=False) as stored:
        actual = stored["signal"][list(SLOTS)][:, :, TEST]
        oracle_variance = stored["variance"][list(SLOTS)]
    variance_db = 10 * np.log10(variance / oracle_variance)
    power = np.mean(abs(actual)**2, axis=-1).clip(1e-18)
    results = {}
    for name, estimated in waves.items():
        error = np.mean(abs(estimated - actual)**2, axis=-1) / power
        summary = {}
        for family in FAMILIES:
            select = [i for i, label in enumerate(LABELS)
                      if label.split("/")[0] == family and not label.endswith("/double_power")]
            summary[family] = source.parent.metrics(error[select], variance_db[select])
        results[name] = dict(families=summary, cases={
            label: source.parent.metrics(errors, noise)
            for label, errors, noise in zip(LABELS, error, variance_db)})
    activation = {label: dict(fraction=float(np.mean(row > 0)), weight=float(row.mean()))
                  for label, row in zip(LABELS, weight)}
    write(root / "scores.json", dict(status="SCORED", oracle_last=True, target_reads=0,
          seed=source.SEED, packets=SHAPE[1], labels=list(LABELS), threshold=float(threshold),
          results=results, activation=activation))


def dispatch(runtime):
    rt = read(runtime)
    root, parent = Path(rt["run_root"]), Path(rt["parent"])
    config, penalty = rt["configuration"], rt["confidence_threshold"]
    stages = (("smoke", (config, penalty)),
              ("predict", (parent / "paired.npz", config, penalty, root / "estimates.npz")),
              ("score", (root, parent / "reference.npz", penalty)))
    source.parent.parent.prior_runner.dispatch(rt, stages)


def publish(repo, output, context, ssh_config, threshold):
    ctx = read(context)
    base = PurePosixPath(ctx["project"])
    runs = base / "runs"
    parent, config = runs / source.RUN, runs / source.parent.RUN / "configuration.json"
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN,
              runner=__spec__.name, code_commit=transport.release_commit(repo),
              release_root=str(base / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              parent=str(parent), configuration=str(config), confidence_threshold=float(threshold),
              external_releases=[str(parent / "paired.npz"), str(config)])
    transport.release(repo, output, rt, ssh_config)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    publish_cmd = commands.add_parser("publish")
    for arg in ("repo", "output", "context", "ssh-config", "threshold"):
        publish_cmd.add_argument("--" + arg, type=float if arg == "threshold" else Path, required=True)
    stages = ((smoke, ("config", "threshold")), (predict, ("paired", "config", "threshold", "output")),
              (score, ("root", "reference", "threshold")), (dispatch, ("runtime",)))
    for function, fields in stages:
        command = commands.add_parser(function.__name__)
        for field in fields:
            command.add_argument(field, type=float if field == "threshold" else Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    {"publish": publish, **{function.__name__: function for function, _ in stages}}[action](**args)


if __name__ == "__main__":
    main()
