"""Validate independent path motion on WiSig source packets outside the earlier panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path, PurePosixPath
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_action_noise_audit as practical
from . import wisig_path_motion as path
from . import wisig_refined_channel_audit as parent
from . import wisig_ten_channel_panel as panel
from .wisig_independent_channel_audit import N, TEST


SEED = 2026112510
RUN = "20261125-diagnostic-wisig-path-motion-s2026112510-r01"
METHODS = ("adaptive", "refined", "path_motion")
PENALTIES = (3.5, 5., 7.)
MIXES = (0., .35, .7)
WORKERS = 4


def smoke():
    parent.smoke()
    rng = np.random.default_rng(37)
    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    shift = path.paired.interpolator(x)
    u = path.U[:, 0]
    y = (.85 - .32j) * (1. - .2 * u) * np.exp(1.7j * u) * shift(np.arange(N) - 18.3)
    y += (.55 + .11j) * (1. + .3 * u) * np.exp(-1.4j * u) * shift(np.arange(N) + 2.1)
    baseline = x
    bank = path.estimate_bank(x, y, baseline, np.arange(N))
    error = min(np.sum(abs(y - entry.action)**2) for entry in bank) / np.sum(abs(y)**2)
    if error >= .001:
        raise ValueError("Independent path-motion smoke failed")
    print("PATH_MOTION_NO_QUERY_SMOKE_PASS", flush=True)


def tune_packet(job):
    x, y, prior = job
    held = np.flatnonzero((np.arange(N) // 8) % 4 == 1)
    train = np.setdiff1d(np.arange(N), held)
    adaptive, refined, _, bank = path.packet(x, y, prior, train)
    candidates = path.paired.solutions(bank, len(train), PENALTIES)
    baseline_loss = np.sum(np.abs(y[held] - adaptive[held])**2).clip(1e-18)
    return [[np.sum(np.abs(y[held] - refined[held] - mix * (wave[held] - refined[held]))**2)
             / baseline_loss for mix in MIXES] for wave in candidates.values()]


def tune(dev, config, output):
    prior = read(config)
    with np.load(dev, allow_pickle=False) as data:
        x, y = data["clean"], data["received"]
    if x.shape != (len(panel.FAMILIES), panel.NDEV, N) or y.shape != x.shape:
        raise ValueError("Expected anonymous ten-group source-U paired IQ")
    jobs = ((a, b, prior) for a, b in zip(x.reshape(-1, N), y.reshape(-1, N)))
    with ProcessPoolExecutor(WORKERS) as pool:
        loss = np.asarray(list(pool.map(tune_packet, jobs, chunksize=8)))
    loss = loss.reshape(len(panel.FAMILIES), panel.NDEV, len(PENALTIES), len(MIXES)).mean(1)
    trace = []
    for i, penalty in enumerate(PENALTIES):
        for j, mix in enumerate(MIXES):
            family = loss[:, i, j]
            trace.append(dict(penalty=penalty, mix=mix, family_ratios=family.tolist(),
                              objective=float(family.max() + family.mean())))
    best = min(trace, key=lambda item: item["objective"])
    write(output, dict(prior=prior, selected=best, candidates=trace))
    print("PATH_MOTION_IQ_CONFIGURATION_FROZEN", best, flush=True)


def fresh_source(source, seed=SEED, excluded_ids=()):
    _, historical_ids, _, _ = practical.source_panel(source)
    root = source / "validation"
    iq, ids, rx, day = [np.load(root / (name + ".npy"), allow_pickle=False)
                        for name in ("clean", "physical_ids", "rx", "day")]
    unseen = ~np.isin(ids, np.r_[historical_ids, np.asarray(excluded_ids, dtype=ids.dtype)])
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(unseen & (rx == r) & (day == d)), 4, replace=False)
        for r, d in product(practical.SOURCE_RX, practical.SOURCE_DAYS)
    ])
    if len(selected) != panel.NTEST or not unseen[selected].all():
        raise ValueError("Unexpected overlap in source-V physical packets")
    clean = iq[selected, 0].astype(np.float64) + 1j * iq[selected, 1]
    sessions = [f"rx{rx[i]}:day{day[i]}:eq1" for i in selected]
    return clean, ids[selected], sessions


def prepare(root, source, seed=SEED, excluded_ids=()):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
    from leo_practical import Config
    from .wisig_dura_stress_data import harder

    clean, ids, sessions = fresh_source(source, seed, excluded_ids)
    observed, signal, variance = [], [], []
    with ProcessPoolExecutor(WORKERS) as pool:
        for profile, route, scene in practical.CELLS:
            cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route=route,
                         mode="post_sync", equalization_enabled=False, boundary="reflect")
            if profile == "strong":
                cfg = harder(cfg)
            jobs = ((cfg, x, ident, session) for x, ident, session in zip(clean, ids, sessions))
            y, h, _, v = map(np.asarray, zip(*pool.map(practical.snapshot, jobs, chunksize=12)))
            observed.append(y)
            signal.append(h)
            variance.append(v)
            print("FRESH_SOURCE_PRACTICAL", profile, route, scene, flush=True)
    extra = (*panel.earlier.ADDITIONAL_TEST,
             *((family, view) for family in panel.EXTRA for view in ("nominal", "stress")))
    for i, (family, view) in enumerate(extra):
        generate = panel.extra_channel if family in panel.EXTRA else panel.earlier.simulate
        y, h, v = generate(clean, family, view, seed + 1000 + i, reference=True)
        observed.append(y)
        signal.append(h)
        variance.append(v)
    received, signal, variance = map(np.asarray, (observed, signal, variance))
    slots = list(parent.doubled.SLOTS)
    strong, doubled = parent.doubled.double_impairment(clean, signal[slots], received[slots])
    np.savez(root / "paired.npz", clean=clean, received=np.concatenate((received, doubled)))
    np.savez(root / "reference.npz", signal=np.concatenate((signal, strong)),
             variance=np.concatenate((variance, 2 * variance[slots])))
    print("FRESH_SOURCE_PAIRED_IQ", len(parent.LABELS), len(clean), flush=True)


def predict_packet(job):
    x, y, fixed = job
    adaptive, refined, variance, bank = path.packet(x, y, fixed["prior"])
    penalty = fixed["selected"]["penalty"]
    wave = path.paired.solutions(bank, N, (penalty,))[penalty]
    estimate = refined + fixed["selected"]["mix"] * (wave - refined)
    return adaptive[TEST], refined[TEST], estimate[TEST], variance


def predict(paired, config, output):
    with np.load(paired, allow_pickle=False) as data:
        clean, received = data["clean"], data["received"]
    if clean.shape != (panel.NTEST, N) or received.shape != (*parent.SHAPE, N):
        raise ValueError("Incomplete paired source-V input")
    options = read(config)
    output_arrays = [np.empty((*parent.SHAPE, len(TEST)), complex) for _ in METHODS]
    variance = np.empty(parent.SHAPE)
    with ProcessPoolExecutor(WORKERS) as pool:
        for i, received_case in enumerate(received):
            jobs = ((x, y, options) for x, y in zip(clean, received_case))
            results = list(pool.map(predict_packet, jobs, chunksize=4))
            for array, column in zip((*output_arrays, variance), zip(*results)):
                array[i] = column
            print("V_ESTIMATES_CLOSED", parent.LABELS[i], len(results), flush=True)
    np.savez(output, **dict(zip(METHODS, output_arrays)), variance=variance)


def score(root):
    parent.score(root, METHODS, SEED, "Zero by physical ID with the earlier archived 360-packet panel")


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt["run_root"])
    config = root / "configuration.json"
    stages = (("smoke", ()), ("tune", (rt["dev"], rt["refined"], config)),
              ("prepare", (root, rt["source"])),
              ("predict", (root / "paired.npz", config, root / "estimates.npz")), ("score", (root,)))
    parent.parent.prior_runner.dispatch(rt, stages)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    base = PurePosixPath(ctx["project"])
    runs = base / "runs"
    inputs = dict(dev=str(runs / parent.parent.RUN / "development.npz"),
                  refined=str(runs / parent.RUN / "configuration.json"),
                  source=str(runs / practical.SOURCE / "cache"))
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN,
              runner=__spec__.name, code_commit=transport.release_commit(repo),
              release_root=str(base / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              external_releases=list(inputs.values()), **inputs)
    transport.release(repo, output, rt, ssh_config)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    release = commands.add_parser("publish")
    for field in ("repo", "output", "context", "ssh-config"):
        release.add_argument("--" + field, type=Path, required=True)
    stages = ((smoke, ()), (tune, ("dev", "config", "output")), (prepare, ("root", "source")),
              (predict, ("paired", "config", "output")), (score, ("root",)), (dispatch, ("runtime",)))
    for function, fields in stages:
        stage = commands.add_parser(function.__name__)
        for field in fields:
            stage.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    selected = args.pop("action")
    {"publish": publish, **{function.__name__: function for function, _ in stages}}[selected](**args)


if __name__ == "__main__":
    main()
