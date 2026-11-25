"""Freeze paired-IQ estimates before comparing them with source simulator traces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_double_impairment as doubled
from . import wisig_refined_channel as refined
from . import wisig_ten_channel_audit as parent
from . import wisig_ten_channel_panel as panel
from .wisig_independent_channel_audit import N, TEST


RUN = "20261125-diagnostic-wisig-channel-refined-s2026112507-r01"
SEED = panel.SEED + 1
PENALTIES = (2., 3.5, 5., 7.)
MIXES = (0., .35, .7, 1.)
METHODS = ("adaptive", "refined")
LABELS = panel.LABELS + tuple(f"{label}/double_power" for label in doubled.LABELS)
SHAPE = (len(LABELS), panel.NTEST)
WORKERS = 4


def smoke():
    rng = np.random.default_rng(7)
    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    interpolate = refined.interpolator(x)
    y = (.8 - .3j) * interpolate(np.arange(N) - 20.25)
    y *= np.exp(1j * (1.5 * refined.adaptive.U + .2 * refined.adaptive.PHASE[:, 1]))
    fixed = {"complexity": 4., "shrink": 0., "blend": .2}
    _, variance, candidates = refined.packet(x, y, fixed)
    prediction = refined.solutions(candidates, N, PENALTIES)[5.]
    if not (np.isfinite(variance) and variance > 0 and
            np.sum(abs(prediction - y)**2) / np.sum(abs(y)**2) < 1e-3):
        raise ValueError("Fractional delay and Doppler smoke failed")
    print("NO_QUERY_SMOKE_PASS", flush=True)


def tune_packet(job):
    x, y, options = job
    train = np.flatnonzero((np.arange(N) // 8) % 4 != 1)
    held = np.flatnonzero((np.arange(N) // 8) % 4 == 1)
    baseline, _, bank = refined.packet(x, y, options, train)
    waves = refined.solutions(bank, len(train), PENALTIES)
    target = y[held] - baseline[held]
    base_power = np.sum(np.abs(target)**2).clip(1e-18)
    return [[np.sum(np.abs(target - mix * (waves[penalty][held] - baseline[held]))**2)
             / base_power for mix in MIXES] for penalty in PENALTIES]


def tune(development, old_config, output):
    with np.load(development, allow_pickle=False) as data:
        clean, received = data["clean"], data["received"]
    if clean.shape != (len(panel.FAMILIES), panel.NDEV, N) or received.shape != clean.shape:
        raise ValueError("Expected ten anonymous source-U paired groups")
    old = read(old_config)
    options = {key: old[key] for key in ("complexity", "shrink", "blend")}
    jobs = [(x, y, options) for x, y in zip(clean.reshape(-1, N), received.reshape(-1, N))]
    with ProcessPoolExecutor(WORKERS) as pool:
        ratios = np.asarray(list(pool.map(tune_packet, jobs, chunksize=8)))
    grouped = ratios.reshape(len(panel.FAMILIES), panel.NDEV, len(PENALTIES), len(MIXES))
    trace = []
    for i, penalty in enumerate(PENALTIES):
        for j, mix in enumerate(MIXES):
            losses = grouped[:, :, i, j].mean(axis=1)
            trace.append(dict(penalty=penalty, mix=mix, family_ratios=losses.tolist(),
                              objective=float(losses.max() + losses.mean())))
    best = min(trace, key=lambda item: item["objective"])
    write(output, dict(adaptive=options, penalty=best["penalty"], mix=best["mix"],
                       source_u_loss=best, candidates=trace))
    print("IQ_ONLY_CONFIGURATION_FROZEN", best, flush=True)


def prepare(root, source_panel):
    panel.challenge(root, source_panel, SEED)
    with np.load(root / "challenge.npz", allow_pickle=False) as staged:
        clean, received = staged["clean"], staged["received"]
    with np.load(root / "oracle.npz", allow_pickle=False) as scorer_only:
        signal, power = scorer_only["signal"], scorer_only["variance"]
    slots = list(doubled.SLOTS)
    strong, observed = doubled.double_impairment(clean, signal[slots], received[slots])
    np.savez(root / "paired.npz", clean=clean, received=np.concatenate((received, observed)))
    np.savez(root / "reference.npz", signal=np.concatenate((signal, strong)),
             variance=np.concatenate((power, 2 * power[slots])))
    print("PAIRED_SOURCE_IQ_READY", len(LABELS), len(clean), flush=True)


def predict_packet(job):
    x, y, fixed = job
    baseline, variance, bank = refined.packet(x, y, fixed["adaptive"])
    penalty = fixed["penalty"]
    proposed = refined.solutions(bank, N, (penalty,))[penalty]
    mixed = baseline + fixed["mix"] * (proposed - baseline)
    return baseline[TEST], mixed[TEST], variance


def predict(paired, config, output):
    with np.load(paired, allow_pickle=False) as data:
        clean, received = data["clean"], data["received"]
    if clean.shape != (panel.NTEST, N) or received.shape != (*SHAPE, N):
        raise ValueError("Unexpected source-V paired IQ shape")
    fixed = read(config)
    actions = {name: np.empty((*SHAPE, len(TEST)), complex) for name in METHODS}
    variance = np.empty(SHAPE)
    with ProcessPoolExecutor(WORKERS) as pool:
        for i, view in enumerate(received):
            rows = list(pool.map(predict_packet, ((x, y, fixed) for x, y in zip(clean, view)),
                                 chunksize=4))
            actions["adaptive"][i], actions["refined"][i], variance[i] = map(np.stack, zip(*rows))
            print("V_ESTIMATES_CLOSED", LABELS[i], len(rows), flush=True)
    if not all(np.isfinite(x).all() for x in (*actions.values(), variance)) or (variance <= 0).any():
        raise ValueError("Nonfinite estimate or nonpositive variance")
    np.savez(output, **actions, variance=variance)


def metrics(error, db):
    return dict(parent.summary(error, db), action_mean=float(error.mean()),
                fraction_under_0p1=float(np.mean(error < .001)))


def score(root, methods=METHODS, seed=SEED,
          overlap="Same 60 physical packets as the prior three-slot Practical audit"):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use an independent CPU scorer")
    with np.load(root / "estimates.npz", allow_pickle=False) as data:
        actions = {name: data[name] for name in methods}
        variance = data["variance"]
    if (variance.shape != SHAPE or (variance <= 0).any() or not np.isfinite(variance).all()
            or any(x.shape != (*SHAPE, len(TEST)) or not np.isfinite(x).all()
                   for x in actions.values())):
        raise ValueError("Both complete estimates are required before loading the reference")
    with np.load(root / "reference.npz", allow_pickle=False) as reference:
        expected, true_v = reference["signal"], reference["variance"]
    if expected.shape != (*SHAPE, N) or true_v.shape != SHAPE:
        raise ValueError("Unexpected scorer reference shape")
    db = 10. * np.log10(variance / true_v)
    target = expected[:, :, TEST]
    power = np.mean(np.abs(target)**2, axis=-1).clip(1e-18)
    # Optimistic in-model benchmark: known exact shape, unknown complex gain,
    # proper white noise and N samples. Squared relative gain error is exponential.
    single_gain_median = np.log(2.) * true_v / np.sum(np.abs(expected)**2, axis=-1).clip(1e-18)
    columns = np.arange(len(LABELS))
    matrices = {
        "original_44": columns[:len(panel.LABELS)],
        "default_matched": np.asarray(doubled.SLOTS[:-1]),
        "double_matched": columns[len(panel.LABELS):-1],
    }
    results = {}
    for name, prediction in actions.items():
        error = np.mean(np.abs(prediction - target)**2, axis=-1) / power
        cases = {case: metrics(error[i], db[i]) for i, case in enumerate(LABELS)}
        families = {}
        for matrix, indices in matrices.items():
            families[matrix] = {}
            for family in panel.FAMILIES:
                include = [i for i in indices if LABELS[i].startswith(family + "/")]
                families[matrix][family] = dict(
                    metrics=metrics(error[include], db[include]),
                    ideal_gain_median=float(np.median(single_gain_median[include])),
                )
        results[name] = dict(families=families, cases=cases)
    write(root / "scores.json", dict(
        status="SCORED", oracle_last=True, target_reads=0, packets=panel.NTEST,
        seed=seed, cases=len(LABELS), threshold=.001, previous_source_v_overlap=overlap,
        results=results, gain_reference="complex-gain LS, known shape, white-noise model",
    ))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt["run_root"])
    config = root / "configuration.json"
    stages = (
        ("smoke", ()), ("tune", (rt["dev"], rt["baseline"], config)),
        ("prepare", (root, rt["panel"])),
        ("predict", (root / "paired.npz", config, root / "estimates.npz")),
        ("score", (root,)),
    )
    parent.prior_runner.dispatch(rt, stages)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    base = PurePosixPath(ctx["project"])
    runs = base / "runs"
    previous = runs / parent.RUN
    source_panel = runs / parent.prior_runner.reference.PANEL
    inputs = dict(dev=str(previous / "development.npz"), baseline=str(previous / "configuration.json"),
                  panel=str(source_panel))
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
    stages = ((smoke, ()), (tune, ("development", "old_config", "output")),
              (prepare, ("root", "source_panel")), (predict, ("paired", "config", "output")),
              (score, ("root",)), (dispatch, ("runtime",)))
    for func, fields in stages:
        stage = commands.add_parser(func.__name__)
        for field in fields:
            stage.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    function = args.pop("action")
    {"publish": publish, **{func.__name__: func for func, _ in stages}}[function](**args)


if __name__ == "__main__":
    main()
