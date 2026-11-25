"""Freeze a Low Urban IQ-only posterior and score its center and tail errors."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_joint_sparse as joint
from . import wisig_joint_sparse_full_audit as full
from . import wisig_low_urban_audit as training
from . import wisig_low_urban_marginal as solver
from .wisig_independent_channel_audit import N, TEST


RUN = "20261125-diagnostic-wisig-low-urban-marginal-s2026112511-r01"
METHODS = ("joint_sparse", "low_urban_marginal")
LABELS = tuple(label for label in full.source.parent.LABELS if "/practical_low_urban" in label)
SLOTS = tuple(full.source.parent.LABELS.index(label) for label in LABELS)
SHAPE = (len(LABELS), full.source.panel.NTEST)
WORKERS = 4
RULES = ((0., 1., 0.), (.5, 1., 0.), (.6, 1., .5), (.9, 1., .5), (.9, 1., 1.), (1., 1., 2.))
NOISE_MIX = (0., .25, .5)


def smoke(config):
    full.smoke(config)
    rng = np.random.default_rng(721)
    x = (rng.normal(size=N) + 1j * rng.normal(size=N)) / np.sqrt(2.)
    tap = 34
    gain = .84 - .27j
    truth = gain * x[solver.OFFSETS[:, tap]]
    noise = .0002
    y = truth + np.sqrt(noise / 2.) * (rng.normal(size=N) + 1j * rng.normal(size=N))
    cov = np.zeros((solver.OFFSETS.shape[1], solver.RANK))
    cov[tap, 0] = 1.
    wave, var = solver.posterior(x, y, noise, solver.TIME, cov)
    if not (np.mean(abs(wave - truth)**2) / np.mean(abs(truth)**2) < .001 and var > 0):
        raise ValueError("Low Urban posterior impulse smoke failed")
    print("LOW_URBAN_MARGINAL_NO_QUERY_SMOKE_PASS", flush=True)


def tune_packet(job):
    original, observed, factor, options = job
    trials = []
    for fold in range(4):
        held = joint.FOLDS == fold
        idx = solver.TIME[~held]
        x, y, _ = solver.normalize(original, observed, idx)
        baseline, base_var = solver.baseline_fit(x, y, options, idx)
        residual = y[idx] - baseline[idx]
        variance = max(float(np.median(abs(residual)**2)) / np.log(2.), .002)
        extra, fitted = solver.posterior(x, y, variance, idx, factor)
        response = y[held] - baseline[held]
        scale = max(float(np.mean(abs(response)**2)), 1e-15)
        risk, nll = [], []
        for rule in RULES:
            amount, quality = solver.mix_weight(y, variance, idx, rule)
            error = float(np.mean(abs(response - amount * (extra - baseline)[held])**2))
            risk.append(error / scale)
            values = [max((1 - mix) * base_var + mix * fitted, 1e-12) for mix in NOISE_MIX]
            nll.append([np.log(v) + error / v for v in values])
        trials.append((np.asarray(risk), np.asarray(nll), quality))
    return tuple(np.asarray(part).mean(axis=0) for part in zip(*trials))


def tune(development, config, output):
    options = read(config)
    with np.load(development, allow_pickle=False) as stored:
        clean, received = stored["clean"], stored["received"]
    if clean.shape != (training.NDEV, N) or received.shape != clean.shape:
        raise ValueError("Expected the frozen source-U Low Urban learning bank")
    fit, hold = np.split(np.random.default_rng(803).permutation(len(clean)), 2)
    factor = solver.fit_covariance(clean[fit], received[fit], options)
    with ProcessPoolExecutor(WORKERS) as pool:
        jobs = ((clean[i], received[i], factor, options) for i in hold)
        rows = list(pool.map(tune_packet, jobs, chunksize=4))
    risks, nll, quality = map(np.stack, zip(*rows))
    weak = np.argsort(quality)[:len(quality) // 4]
    tail = np.quantile(risks, [.9, .95], axis=0)
    score = risks.mean(axis=0) + risks[weak].mean(axis=0) + tail.mean(axis=0)
    pick = int(np.argmin(score))
    weighted_nll = nll[:, pick] - nll[:, pick, :1]
    vscore = weighted_nll.mean(axis=0) + np.quantile(weighted_nll, .95, axis=0)
    variance_pick = int(np.argmin(vscore))
    write(output, dict(rule=RULES[pick], variance_mix=NOISE_MIX[variance_pick],
          wave_validation=[dict(rule=rule, score=float(s), mean=float(mean), p90=float(p90), p95=float(p95))
                           for rule, s, mean, p90, p95 in zip(RULES, score, risks.mean(0), *tail)],
          variance_validation=vscore.tolist()))
    np.save(output.with_suffix(".npy"), factor, allow_pickle=False)
    print("SOURCE_U_LOW_URBAN_FROZEN", RULES[pick], NOISE_MIX[variance_pick], len(fit), len(hold), flush=True)


def predict_packet(job):
    clean, observed, prior, options, fixed = job
    _, reference, variance, _ = joint.estimate(clean, observed, options, full.THRESHOLD)
    x, y, level = solver.normalize(clean, observed)
    normalized_var = max(variance / level**2, .002)
    extra, adjusted_var = solver.posterior(x, y, normalized_var, solver.TIME, prior)
    weight, _ = solver.mix_weight(y, normalized_var, solver.TIME, fixed["rule"])
    enhanced = reference + weight * (level * extra - reference)
    amount = fixed["variance_mix"]
    enhanced_var = (1. - amount) * variance + amount * adjusted_var * level**2
    return reference[TEST], enhanced[TEST], variance, enhanced_var, weight


def predict(paired, config, fitted, output):
    with np.load(paired, allow_pickle=False) as stored:
        clean, received = stored["clean"], stored["received"]
    if clean.shape != (SHAPE[1], N) or received.shape != (*full.source.parent.SHAPE, N):
        raise ValueError("Expected the registered source-V Low Urban packet pairs")
    options, fixed = read(config), read(fitted)
    factor = np.load(fitted.with_suffix(".npy"), allow_pickle=False)
    if factor.shape != (solver.OFFSETS.shape[1], solver.RANK) or not np.isfinite(factor).all():
        raise ValueError("Expected the source-U channel covariance factor")
    waves = {name: np.empty((*SHAPE, len(TEST)), complex) for name in METHODS}
    noise = {name: np.empty(SHAPE) for name in METHODS}
    weights = np.empty(SHAPE)
    with ProcessPoolExecutor(WORKERS) as pool:
        for i, slot in enumerate(SLOTS):
            jobs = ((x, y, factor, options, fixed) for x, y in zip(clean, received[slot]))
            rows = list(pool.map(predict_packet, jobs, chunksize=3))
            waves[METHODS[0]][i], waves[METHODS[1]][i], noise[METHODS[0]][i], noise[METHODS[1]][i], weights[i] = zip(*rows)
            print("LOW_URBAN_ESTIMATES_CLOSED", LABELS[i], len(rows), flush=True)
    np.savez(output, **waves, **{name + "_variance": v for name, v in noise.items()}, weight=weights)


def metrics(values):
    p50, p90, p95 = np.quantile(values, [.5, .90, .95])
    return dict(mean=float(values.mean()), median=float(p50), p90=float(p90), p95=float(p95),
                maximum=float(values.max()))


def score(root, reference):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use a separate CPU scorer")
    with np.load(root / "estimates.npz", allow_pickle=False) as stored:
        predictions = {name: (stored[name], stored[name + "_variance"]) for name in METHODS}
        weights = stored["weight"]
    if (weights.shape != SHAPE or not ((0. <= weights) & (weights <= 1.)).all() or
        any(w.shape != (*SHAPE, len(TEST)) or v.shape != SHAPE or (v <= 0).any() or
            not np.isfinite(v).all() or not np.isfinite(w).all() for w, v in predictions.values())):
        raise ValueError("Freeze both Low Urban predictions before the independent score")
    with np.load(reference, allow_pickle=False) as stored:
        actual, true_noise = stored["signal"][list(SLOTS)][:, :, TEST], stored["variance"][list(SLOTS)]
    power = np.mean(abs(actual)**2, axis=-1).clip(1e-18)
    results = {}
    raw_errors = []
    for name, (wave, variance) in predictions.items():
        error = np.mean(abs(wave - actual)**2, axis=-1) / power
        raw_errors.append(error)
        noise_error = abs(10. * np.log10(variance / true_noise))
        results[name] = {case: dict(waveform_nmse=metrics(e), noise_abs_db=metrics(n))
                         for case, e, n in zip(LABELS, error, noise_error)}
    bootstrap = np.random.default_rng(4693).integers(SHAPE[1], size=(1200, SHAPE[1]))
    confidence = {}
    for i, case in enumerate(LABELS):
        old, new = (value[i, bootstrap] for value in raw_errors)
        diff = np.quantile(new, [.5, .9, .95], axis=1) - np.quantile(old, [.5, .9, .95], axis=1)
        confidence[case] = np.quantile(diff, [.025, .975], axis=1).T.tolist()
    write(root / "scores.json", dict(status="SCORED", oracle_last=True, target_reads=0,
          seed=full.SEED, packets=SHAPE[1], labels=list(LABELS), results=results,
          average_weight=dict(zip(LABELS, weights.mean(axis=1).tolist())), paired_intervals=confidence))


def dispatch(runtime):
    rt = read(runtime)
    root, parent = Path(rt["run_root"]), Path(rt["parent"])
    cfg, fitted = rt["configuration"], root / "configuration.json"
    stages = (("smoke", (cfg,)), ("tune", (rt["development"], cfg, fitted)),
              ("predict", (parent / "paired.npz", cfg, fitted, root / "estimates.npz")),
              ("score", (root, parent / "reference.npz")))
    full.source.parent.parent.prior_runner.dispatch(rt, stages)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    parent = runs / full.RUN
    config = runs / full.source.parent.RUN / "configuration.json"
    development = runs / training.RUN / "development.npz"
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN,
              runner=__spec__.name, code_commit=transport.release_commit(repo),
              release_root=str(runs.parent / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              parent=str(parent), configuration=str(config), development=str(development),
              external_releases=[str(parent / "paired.npz"), str(config), str(development)])
    transport.release(repo, output, rt, ssh_config)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    pub = commands.add_parser("publish")
    for arg in ("repo", "output", "context", "ssh-config"):
        pub.add_argument("--" + arg, type=Path, required=True)
    stages = ((smoke, ("config",)), (tune, ("development", "config", "output")),
              (predict, ("paired", "config", "fitted", "output")), (score, ("root", "reference")),
              (dispatch, ("runtime",)))
    for function, fields in stages:
        stage = commands.add_parser(function.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    {"publish": publish, **{function.__name__: function for function, _ in stages}}[action](**args)


if __name__ == "__main__":
    main()
