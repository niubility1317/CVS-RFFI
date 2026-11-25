"""Calibrate on source U; score only Practical views after observed-IQ prediction."""

import argparse
import os
from pathlib import Path, PurePosixPath
import sys
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_joint_sparse as joint
from . import wisig_joint_sparse_full_audit as full
from . import wisig_low_urban_marginal_audit as urban
from . import wisig_practical_fast as model
from .wisig_dura_stress_data import harder
from .wisig_independent_channel_audit import N, TEST
from .wisig_lora_source_fit import source_iq
from .wisig_paired_estimator import voltages


RUN = "20261126-diagnostic-wisig-practical-fast-s2026112512-r01"
SEED = 2026112512
LABELS = tuple(label for label in full.source.parent.LABELS if label.startswith("practical/"))
SLOTS = tuple(full.source.parent.LABELS.index(label) for label in LABELS)
SHAPE = (len(LABELS), full.source.panel.NTEST)
NDEV = 144
RANKS = (6, 10, 14, 18)
CONCENTRATIONS = (.35, 1., 2.5)
NOISE_OPTIONS = ((0., .97), (0., 1.), (1., .97), (1., 1.), (1., 1.03))
DEVELOPMENT_CELLS = tuple((profile, scene) for profile, route, scene in full.source.practical.CELLS if route == "full")
TARGETS = dict(mean=.001, p90=.01, p95=.005, variance_median=.1, variance_p95=.5)


def smoke():
    rng = np.random.default_rng(712)
    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    anchor = int(np.flatnonzero(model.LAGS == 24)[0])
    clean = .65 * x[model.OFFSETS[:, anchor]]
    noisy = clean + .01 * (rng.normal(size=N) + 1j * rng.normal(size=N))
    factor = np.zeros((len(model.LAGS), model.MAX_RANK))
    factor[anchor, 0] = 1.
    bank = model.fit_components(x, noisy, model.TIME, factor, anchor, 6)
    result, noise = model.posterior(bank, 2.5, 1.)
    if not (np.mean(abs(result - clean)**2) / np.mean(abs(clean)**2) < .001 and
            1e-5 < noise < .001):
        raise ValueError("Fast Practical synthetic paired-IQ smoke failed")
    print("PRACTICAL_FAST_NO_QUERY_SMOKE_PASS", flush=True)


def prepare(root, cache):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
    from leo_practical import ChannelStream, Config, receiver_for_session, stable_seed

    rows, allowed, _ = source_iq(cache)
    rng = np.random.default_rng(SEED)
    selected = rng.choice(allowed, NDEV, replace=False)
    x = voltages(rows[selected])
    columns = [np.load(cache / "train" / (key + ".npy"), mmap_mode="r", allow_pickle=False)[selected]
               for key in ("physical_ids", "rx", "day")]
    observed = []
    for profile, scene in DEVELOPMENT_CELLS:
        cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route="full",
                     mode="post_sync", equalization_enabled=False, boundary="reflect")
        if profile == "strong":
            cfg = harder(cfg)
        stack = []
        for clean, ident, rx, day in zip(x, *columns):
            seed = stable_seed(SEED, "fast_practical/source_U", scene, str(ident))
            receiver = receiver_for_session(cfg, SEED + 1, f"rx{rx}:day{day}:eq1")
            stack.append(ChannelStream(cfg, seed, receiver).process(clean)[0])
        observed.append(stack)
    np.savez(root / "development.npz", clean=x, received=observed)
    print("PRACTICAL_SOURCE_U_READY", len(observed), len(x), flush=True)


def tune(development, fitted, base_config):
    with np.load(development, allow_pickle=False) as stored:
        clean, received = stored["clean"], stored["received"]
    if clean.shape != (NDEV, N) or received.shape != (len(DEVELOPMENT_CELLS), NDEV, N):
        raise ValueError("Expected the registered source-U development bank")
    rng = np.random.default_rng(SEED)
    fit, held = np.split(rng.permutation(len(clean)), [96])
    subsets = [rng.choice(fit, 28, replace=False) for _ in received]
    factor, anchor = model.learn(np.concatenate([clean[s] for s in subsets]),
                                 np.concatenate([y[s] for y, s in zip(received, subsets)]))
    actions = [(rank, concentration) for rank in RANKS for concentration in CONCENTRATIONS]
    risk = np.empty((len(received), len(held), len(actions), 2, len(NOISE_OPTIONS) + 1))
    for g, view in enumerate(received):
        for j, index in enumerate(held):
            x, y = clean[index], view[index]
            for fold in range(2):
                test = model.TIME // 8 % 2 == fold
                train = model.TIME[~test]
                scale = max(float(np.mean(abs(y[train])**2)), 1e-12)
                banks = {r: model.fit_components(x, y, train, factor, anchor, r) for r in RANKS}
                for k, (rank, concentration) in enumerate(actions):
                    wave, posterior = model.posterior(banks[rank], concentration, 1.)
                    base = banks[rank][-2] * banks[rank][-1]**2
                    error = float(np.mean(abs(y[test] - wave[test])**2))
                    noise = [multiplier * ((1 - shrink) * base + shrink * posterior)
                             for shrink, multiplier in NOISE_OPTIONS]
                    risk[g, j, k, fold] = [error / scale, *[np.log(v / scale) + error / v for v in noise]]
    action = risk[..., 0].mean(3)
    objective = action.mean(axis=(0, 1)) + np.quantile(action, .9, axis=1).mean(axis=0)
    choice = int(np.argmin(objective))
    rank, concentration = actions[choice]
    noise_risk = risk[:, :, choice, :, 1:].mean(axis=(0, 1, 2))
    shrink, multiplier = NOISE_OPTIONS[int(np.argmin(noise_risk))]
    settings = dict(anchor=anchor, anchor_delay=int(model.LAGS[anchor]), rank=rank,
                    concentration=concentration, shrink=shrink, variance_multiplier=multiplier,
                    observed_holdout_wave_risk=objective.tolist(), observed_holdout_noise_risk=noise_risk.tolist())
    write(fitted, settings)
    np.save(fitted.with_suffix(".npy"), factor, allow_pickle=False)
    benchmark(fitted.parent, clean[held], received, held, factor, settings, read(base_config))
    print("FROZEN_PRACTICAL_FAST", settings["anchor_delay"], rank, concentration, shrink, multiplier, flush=True)


def estimate(x, y, factor, settings):
    parts = model.fit_components(x, y, model.TIME, factor, settings["anchor"], settings["rank"])
    wave, noise = model.posterior(parts, settings["concentration"], settings["shrink"])
    return wave, settings["variance_multiplier"] * noise


def benchmark(root, clean, received, held, factor, settings, base_config):
    groups = (0, 2, 4, 5, 6, 8, 10, 11)
    rows = [(clean[j], received[g, held[j]]) for j, g in enumerate(groups)]
    estimate(*rows[0], factor, settings)
    joint.estimate(*rows[0], base_config, full.THRESHOLD)
    timings = []
    for x, y in rows:
        start = perf_counter()
        joint.estimate(x, y, base_config, full.THRESHOLD)
        old = perf_counter() - start
        start = perf_counter()
        for _ in range(24):
            estimate(x, y, factor, settings)
        timings.append((old * 1000, (perf_counter() - start) * 1000 / 24))
    values = np.asarray(timings)
    write(root / "speed.json", dict(packets=len(rows), unit="ms/packet", threads=1,
                                   baseline_median=float(np.median(values[:, 0])),
                                   candidate_median=float(np.median(values[:, 1])),
                                   baseline_p95=float(np.quantile(values[:, 0], .95)),
                                   candidate_p95=float(np.quantile(values[:, 1], .95))))


def predict(paired, fitted, output):
    with np.load(paired, allow_pickle=False) as stored:
        clean, received = stored["clean"], stored["received"]
    if clean.shape != (SHAPE[1], N) or received.shape != (*full.source.parent.SHAPE, N):
        raise ValueError("Expected the registered Practical comparison pair bank")
    settings = read(fitted)
    factor = np.load(fitted.with_suffix(".npy"), allow_pickle=False)
    waves, powers = [], []
    for i, slot in enumerate(SLOTS):
        rows = [estimate(x, y, factor, settings) for x, y in zip(clean, received[slot])]
        y, v = map(np.asarray, zip(*rows))
        waves.append(y[:, TEST])
        powers.append(v)
        print("PRACTICAL_ESTIMATES_CLOSED", LABELS[i], len(rows), flush=True)
    np.savez(output, wave=waves, variance=powers)


def metrics(wave_error, variance_error):
    return dict(mean=float(wave_error.mean()), median=float(np.median(wave_error)),
                p90=float(np.quantile(wave_error, .9)), p95=float(np.quantile(wave_error, .95)),
                variance_median=float(np.median(variance_error)), variance_p95=float(np.quantile(variance_error, .95)))


def score(root, parent, low):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use a separate CPU scorer")
    with np.load(root / "estimates.npz", allow_pickle=False) as f:
        candidate, variance = f["wave"], f["variance"]
    if (candidate.shape != (*SHAPE, len(TEST)) or variance.shape != SHAPE or not np.isfinite(candidate).all()
            or not np.isfinite(variance).all() or (variance <= 0).any()):
        raise ValueError("Freeze the entire Practical prediction matrix before reading reference")
    with np.load(parent / "estimates.npz", allow_pickle=False) as f:
        baseline, baseline_var = f["joint_sparse"][list(SLOTS)], f["variance"][list(SLOTS)]
    with np.load(low / "estimates.npz", allow_pickle=False) as f:
        low_indices = [LABELS.index(label) for label in urban.LABELS]
        baseline[low_indices], baseline_var[low_indices] = f["low_urban_marginal"], f["low_urban_marginal_variance"]
    with np.load(parent / "reference.npz", allow_pickle=False) as f:
        signal, true_variance = f["signal"][list(SLOTS)][:, :, TEST], f["variance"][list(SLOTS)]
    power = np.mean(abs(signal)**2, axis=-1).clip(1e-18)
    native = [i for i, label in enumerate(LABELS) if not label.endswith("/double_power")]
    groups = dict(native=native,
                  default=[i for i in native if "/default/" in LABELS[i]],
                  strong=[i for i in native if "/strong/" in LABELS[i]],
                  doubled=list(range(len(native), len(LABELS))),
                  low_urban=[i for i in native if "/practical_low_urban" in LABELS[i]])
    results = {}
    for name, prediction, noise in (("current", baseline, baseline_var), ("practical_fast", candidate, variance)):
        e = np.mean(abs(prediction - signal)**2, axis=-1) / power
        v = abs(10. * np.log10(noise / true_variance))
        summary = {name: metrics(e[take], v[take]) for name, take in groups.items()}
        results[name] = dict(summary=summary, cases={label: metrics(a, b) for label, a, b in zip(LABELS, e, v)})
    aggregate = results["practical_fast"]["summary"]["native"]
    write(root / "scores.json", dict(status="SCORED", oracle_last=True, target_reads=0, cases=len(LABELS),
          packets=SHAPE[1], labels=list(LABELS), thresholds=TARGETS,
          passed={k: aggregate[k] < limit for k, limit in TARGETS.items()}, results=results,
          speed=read(root / "speed.json")))


def dispatch(runtime):
    rt = read(runtime)
    root, parent = Path(rt["run_root"]), Path(rt["parent"])
    fitted = root / "configuration.json"
    stages = (("smoke", ()), ("prepare", (root, rt["cache"])),
              ("tune", (root / "development.npz", fitted, rt["baseline"])),
              ("predict", (parent / "paired.npz", fitted, root / "estimates.npz")),
              ("score", (root, parent, rt["low"])))
    full.source.parent.parent.prior_runner.dispatch(rt, stages)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(parent=str(runs / full.RUN), low=str(runs / urban.RUN),
                  cache=str(runs / full.source.practical.SOURCE / "cache"),
                  baseline=str(runs / full.source.parent.RUN / "configuration.json"))
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN,
              runner=__spec__.name, code_commit=transport.release_commit(repo),
              release_root=str(runs.parent / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              external_releases=list(inputs.values()), **inputs)
    transport.release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="action", required=True)
    pub = sub.add_parser("publish")
    for arg in ("repo", "output", "context", "ssh-config"):
        pub.add_argument("--" + arg, type=Path, required=True)
    stages = ((smoke, ()), (prepare, ("root", "cache")), (tune, ("development", "fitted", "base_config")),
              (predict, ("paired", "fitted", "output")), (score, ("root", "parent", "low")),
              (dispatch, ("runtime",)))
    for function, fields in stages:
        stage = sub.add_parser(function.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    option = args.pop("action")
    {"publish": publish, **{function.__name__: function for function, _ in stages}}[option](**args)


if __name__ == "__main__":
    main()
