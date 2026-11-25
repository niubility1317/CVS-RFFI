"""Source-U selection and independent Practical-only scoring of a prior mixture."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_fast_audit as fast
from . import wisig_practical_mixture as model


RUN = "20261126-diagnostic-wisig-practical-mixture-s2026112512-r01"
OPTIONS = tuple((a,b) for a in (0., .7, 1.3) for b in (0., .65, 1.))
VARIANCE_FACTORS = (.97, 1., 1.03)


def smoke():
    fast.smoke()
    rng = np.random.default_rng(910)
    x = rng.normal(size=fast.N) + 1j * rng.normal(size=fast.N)
    anchor = int(np.flatnonzero(fast.model.LAGS == 24)[0])
    waveform = .6 * x[fast.model.OFFSETS[:, anchor]]
    observed = waveform + .02 * (rng.normal(size=fast.N) + 1j * rng.normal(size=fast.N))
    factor = np.zeros((len(model.GROUPS), len(fast.model.LAGS), fast.model.MAX_RANK))
    factor[:, anchor, 0] = 1.
    posterior = model.observed_evidence(x, observed, fast.model.TIME, factor,
                                        [anchor] * len(factor), [np.linspace(-9., -5., 24)] * len(factor))
    signal, noise = model.estimate(posterior, .7, .65)
    if np.mean(abs(signal - waveform)**2) / np.mean(abs(waveform)**2) >= .001 or noise <= 0:
        raise ValueError("Practical mixture synthetic smoke failed")
    print("PRACTICAL_MIXTURE_NO_QUERY_SMOKE_PASS", flush=True)


def tune(development, prior_config, fitted):
    with np.load(development, allow_pickle=False) as stored:
        clean, received = stored["clean"], stored["received"]
    if clean.shape != (fast.NDEV, fast.N) or received.shape != (12, fast.NDEV, fast.N):
        raise ValueError("Expected Practical source-U bank with unchanged condition order")
    prior = read(prior_config)
    static = np.load(prior_config.with_suffix(".npy"), allow_pickle=False)
    train, held = np.split(np.random.default_rng(fast.SEED).permutation(len(clean)), [96])
    factors, anchors, noise = model.train(clean, received, train, fast.SEED + 1)
    risk = np.empty((len(received), len(held), 2, len(OPTIONS), len(VARIANCE_FACTORS) + 1))
    for g, view in enumerate(received):
        for j, index in enumerate(held):
            x, y = clean[index], view[index]
            for fold in range(2):
                test = fast.model.TIME // 8 % 2 == fold
                idx = fast.model.TIME[~test]
                components = fast.model.fit_components(x, y, idx, static, prior["anchor"], prior["rank"])
                base_wave, base_noise = fast.model.posterior(components, prior["concentration"], prior["shrink"])
                base_noise *= prior["variance_multiplier"]
                base_error = max(float(np.mean(abs(y[test] - base_wave[test])**2)), 1e-15)
                base_loss = np.log(base_noise) + base_error / base_noise
                support = model.observed_evidence(x, y, idx, factors, anchors, noise)
                for k, (noise_prior, relative) in enumerate(OPTIONS):
                    wave, variance = model.estimate(support, noise_prior, relative)
                    err = np.mean(abs(y[test] - wave[test])**2)
                    powers = variance * np.asarray(VARIANCE_FACTORS)
                    risk[g, j, fold, k] = [err / base_error, *(np.log(powers) + err / powers - base_loss)]
    data = risk.mean(axis=2)
    wave = data[..., 0]
    action_risk = wave.mean(axis=(0, 1)) + .25 * np.quantile(wave, .95, axis=(0, 1))
    best = int(np.argmin(action_risk))
    variances = data[..., best, 1:].mean(axis=(0, 1))
    n = int(np.argmin(variances))
    settings = dict(noise_prior=OPTIONS[best][0], relative=OPTIONS[best][1],
                    variance_factor=VARIANCE_FACTORS[n], action_risk=action_risk.tolist(),
                    variance_risk=variances.tolist())
    write(fitted, settings)
    np.savez(fitted.with_suffix(".npz"), factors=factors, anchors=anchors,
             **{f"noise_{j}": row for j, row in enumerate(noise)})
    baseline_speed = read(prior_config.parent / "speed.json")
    samples = [(clean[i], received[g, i]) for g, i in zip((0, 2, 4, 5, 6, 8, 10, 11), held)]
    elapsed = []
    for x, y in samples:
        start = perf_counter()
        for _ in range(24):
            estimate(x, y, factors, anchors, noise, settings)
        elapsed.append((perf_counter() - start) * 1000 / 24)
    write(fitted.parent / "speed.json", dict(packets=len(samples), unit="ms/packet", threads=1,
          candidate_median=float(np.median(elapsed)), candidate_p95=float(np.quantile(elapsed, .95)),
          prior_run_joint_sparse_median=baseline_speed["baseline_median"],
          prior_run_first_fast_median=baseline_speed["candidate_median"]))
    print("FROZEN_PRACTICAL_MIXTURE", settings["noise_prior"], settings["relative"], settings["variance_factor"], flush=True)


def estimate(x, y, factors, anchors, noise, cfg):
    bank = model.observed_evidence(x, y, fast.model.TIME, factors, anchors, noise)
    wave, power = model.estimate(bank, cfg["noise_prior"], cfg["relative"])
    return wave, power * cfg["variance_factor"]


def predict(paired, fitted, output):
    with np.load(paired, allow_pickle=False) as stored:
        x, y = stored["clean"], stored["received"]
    if x.shape != (fast.SHAPE[1], fast.N) or y.shape != (*fast.full.source.parent.SHAPE, fast.N):
        raise ValueError("Wrong frozen paired IQ matrix")
    params = read(fitted)
    with np.load(fitted.with_suffix(".npz"), allow_pickle=False) as stored:
        factors, anchors = stored["factors"], stored["anchors"]
        noise = [stored[f"noise_{i}"] for i in range(len(anchors))]
    waves, power = [], []
    for k, slot in enumerate(fast.SLOTS):
        rows = [estimate(a,b,factors,anchors,noise,params) for a,b in zip(x,y[slot])]
        prediction, variance = map(np.asarray, zip(*rows))
        waves.append(prediction[:, fast.TEST])
        power.append(variance)
        print("PRACTICAL_MIXTURE_CLOSED", fast.LABELS[k], len(rows), flush=True)
    np.savez(output, wave=waves, variance=power)


def score(root, parent, low):
    fast.score(root, parent, low, "practical_mixture")


def dispatch(runtime):
    rt = read(runtime)
    root, parent, prior = Path(rt["run_root"]), Path(rt["parent"]), Path(rt["previous"])
    fitted = root / "configuration.json"
    phases = (("smoke", ()), ("tune", (prior / "development.npz", prior / "configuration.json", fitted)),
              ("predict", (parent / "paired.npz", fitted, root / "estimates.npz")),
              ("score", (root, parent, rt["low"])))
    fast.full.source.parent.parent.prior_runner.dispatch(rt, phases)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(parent=str(runs / fast.full.RUN), low=str(runs / fast.urban.RUN), previous=str(runs / fast.RUN))
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN, runner=__spec__.name,
              code_commit=transport.release_commit(repo),
              release_root=str(runs.parent / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              external_releases=list(inputs.values()), **inputs)
    transport.release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="action", required=True)
    pub = sub.add_parser("publish")
    for arg in ("repo", "output", "context", "ssh-config"):
        pub.add_argument("--" + arg, type=Path, required=True)
    steps = ((smoke, ()), (tune, ("development", "prior_config", "fitted")),
             (predict, ("paired", "fitted", "output")), (score, ("root", "parent", "low")), (dispatch, ("runtime",)))
    for func, fields in steps:
        stage = sub.add_parser(func.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    name = args.pop("action")
    {"publish": publish, **{f.__name__: f for f, _ in steps}}[name](**args)


if __name__ == "__main__":
    main()
