"""Validate per-packet splitting for the three remaining waveform-error families."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_path_motion_audit as motion
from .wisig_independent_channel_audit import N, TEST


RUN = "20261125-diagnostic-wisig-adaptive-path-s2026112510-r01"
FAMILIES = ("multipath_burst", "fractional_clock", "time_selective")
LABELS = tuple(label for label in motion.parent.LABELS if label.split("/")[0] in FAMILIES)
SLOTS = tuple(motion.parent.LABELS.index(label) for label in LABELS)
SHAPE = (len(LABELS), motion.panel.NTEST)
METHODS = ("refined", "adaptive_path")
FOLDS = np.arange(N) // 8 % 4
PATH_PENALTY = 5.
UNCERTAINTY_THRESHOLD = 2.


def packet_fit(x, y, options, indices):
    _, baseline, variance, bank = motion.path.packet(x, y, options, indices)
    alternative = motion.path.paired.solutions(bank, len(indices), (PATH_PENALTY,))[PATH_PENALTY]
    return baseline, alternative - baseline, variance


def blend_weight(x, y, options):
    numerator, denominator, uncertainty = 0., 0., 0.
    for fold in range(4):
        hold = FOLDS == fold
        start, delta, _ = packet_fit(x, y, options, np.flatnonzero(~hold))
        residual = y - start
        power = max(float(np.median(abs(residual[~hold])**2)) / np.log(2.), 1e-18)
        residual = residual[hold]
        magnitude = np.abs(residual)
        weight = np.minimum(1., 4. * max(float(np.median(magnitude)), 1e-12)
                            / np.maximum(magnitude, 1e-12))
        direction = delta[hold]
        energy = np.sum(weight * abs(direction)**2)
        numerator += np.vdot(weight * direction, residual).real
        denominator += energy
        uncertainty += .5 * power * energy
    amount = (numerator - UNCERTAINTY_THRESHOLD * np.sqrt(uncertainty)) / max(denominator, 1e-15)
    return float(np.clip(amount, 0., 1.))


def predict_packet(job):
    x, y, options = job
    amount = blend_weight(x, y, options)
    baseline, change, variance = packet_fit(x, y, options, np.arange(N))
    result = baseline + amount * change
    return baseline[TEST], result[TEST], variance, amount


def smoke(config):
    motion.smoke()
    options = read(config)
    rng = np.random.default_rng(811)
    iq = rng.normal(size=N) + 1j * rng.normal(size=N)
    observed = (.8 + .25j) * iq + .015 * (rng.normal(size=N) + 1j * rng.normal(size=N))
    a, b, variance, amount = predict_packet((iq, observed, options))
    if not (np.isfinite(a).all() and np.isfinite(b).all() and variance > 0 and 0. <= amount <= 1.):
        raise ValueError("Adaptive path smoke failed")
    print("ADAPTIVE_PATH_NO_QUERY_SMOKE_PASS", flush=True)


def predict(paired, config, output):
    with np.load(paired, allow_pickle=False) as data:
        x, y = data["clean"], data["received"]
    packets = motion.panel.NTEST
    if x.shape != (packets, N) or y.shape != (*motion.parent.SHAPE, N):
        raise ValueError("Unexpected frozen source V IQ")
    options = read(config)
    waves = {name: np.empty((*SHAPE, len(TEST)), complex) for name in METHODS}
    variance, amounts = np.empty(SHAPE), np.empty(SHAPE)
    with ProcessPoolExecutor(motion.WORKERS) as pool:
        for i, slot in enumerate(SLOTS):
            jobs = ((clean, received, options) for clean, received in zip(x, y[slot]))
            rows = list(pool.map(predict_packet, jobs, chunksize=3))
            waves["refined"][i], waves["adaptive_path"][i], variance[i], amounts[i] = zip(*rows)
            print("V_ESTIMATES_CLOSED", LABELS[i], packets, flush=True)
    np.savez(output, **waves, variance=variance, amount=amounts)


def score(root, reference):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use independent CPU scoring")
    with np.load(root / "estimates.npz", allow_pickle=False) as archive:
        closed = {name: archive[name] for name in METHODS}
        variance, amounts = archive["variance"], archive["amount"]
    if (variance.shape != SHAPE or amounts.shape != SHAPE or (variance <= 0).any()
            or not np.isfinite(variance).all() or not ((0 <= amounts) & (amounts <= 1)).all()
            or any(x.shape != (*SHAPE, len(TEST)) or not np.isfinite(x).all() for x in closed.values())):
        raise ValueError("Freeze both complete estimates before opening the source reference")
    with np.load(reference, allow_pickle=False) as source:
        target, true_variance = source["signal"][list(SLOTS)][:, :, TEST], source["variance"][list(SLOTS)]
    db = 10 * np.log10(variance / true_variance)
    results = {}
    for name, prediction in closed.items():
        error = np.mean(abs(prediction - target)**2, axis=-1)
        error /= np.mean(abs(target)**2, axis=-1).clip(1e-18)
        groups = {}
        for family in FAMILIES:
            original = [i for i, label in enumerate(LABELS)
                        if label.startswith(family + "/") and not label.endswith("/double_power")]
            groups[family] = motion.parent.metrics(error[original], db[original])
        results[name] = dict(families=groups, cases={label: motion.parent.metrics(error[i], db[i])
                                                    for i, label in enumerate(LABELS)})
    amount_by_case = {label: dict(nonzero=float(np.mean(a > 0)), average=float(np.mean(a)))
                      for label, a in zip(LABELS, amounts)}
    write(root / "scores.json", dict(status="SCORED", oracle_last=True, target_reads=0,
                                   seed=motion.SEED, packets=SHAPE[1], labels=list(LABELS),
                                   results=results, weight_by_case=amount_by_case))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt["run_root"])
    source, cfg = Path(rt["parent"]), rt["configuration"]
    stages = (("smoke", (cfg,)), ("predict", (source / "paired.npz", cfg, root / "estimates.npz")),
              ("score", (root, source / "reference.npz")))
    motion.parent.parent.prior_runner.dispatch(rt, stages)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    base = PurePosixPath(ctx["project"])
    runs = base / "runs"
    parent = runs / motion.RUN
    config = runs / motion.parent.RUN / "configuration.json"
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN,
              runner=__spec__.name, code_commit=transport.release_commit(repo),
              release_root=str(base / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              parent=str(parent), configuration=str(config),
              external_releases=[str(parent / "paired.npz"), str(config)])
    transport.release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser()
    commands = cli.add_subparsers(dest="action", required=True)
    release = commands.add_parser("publish")
    for field in ("repo", "output", "context", "ssh-config"):
        release.add_argument("--" + field, type=Path, required=True)
    stages = ((smoke, ("config",)), (predict, ("paired", "config", "output")),
              (score, ("root", "reference")), (dispatch, ("runtime",)))
    for function, fields in stages:
        command = commands.add_parser(function.__name__)
        for field in fields:
            command.add_argument(field, type=Path)
    args = vars(cli.parse_args())
    option = args.pop("action")
    {"publish": publish, **{function.__name__: function for function, _ in stages}}[option](**args)


if __name__ == "__main__":
    main()
