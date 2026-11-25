"""Freeze source-U relative estimates before the matched R07 probe scoring."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_audit as matched
from . import wisig_practical_probe_relative as model


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r02"
ARMS = ("relative_source", "relative_weak_tail")
RIDGES = (12., 40., 120., 400., 1000.)


def load_prior(location):
    with np.load(location, allow_pickle=False) as a:
        values = {k: a[k] for k in matched.model.Prior.__dataclass_fields__}
    values.update(anchor=int(values["anchor"]), phase_mean=complex(values["phase_mean"]))
    return matched.model.Prior(**values)


def tune(source, prior_path, output):
    receive = np.load(source, allow_pickle=False)
    if receive.shape != (12, matched.protocol.panel.NDEV, matched.model.receive.SAMPLES):
        raise ValueError("The matched receive-only source U bank is required")
    prior = load_prior(prior_path)
    indices = np.random.default_rng(matched.protocol.panel.SEED).permutation(receive.shape[1])
    fit, hold = np.split(indices, [96])
    train, test = (receive[:, g].transpose(1, 0, 2).astype(np.complex128) for g in (fit, hold))
    teacher, validation = (model.target_columns(a, prior) for a in (train, test))
    samples = train.reshape(-1, train.shape[-1])
    held = test.reshape(-1, test.shape[-1])
    features, noise = model.observables(samples, prior)
    h, hq = model.observables(held, prior)
    center, scale = features.mean(axis=0), features.std(axis=0).clip(.05)
    design = model.regressors(features, noise, center, scale)
    h = model.regressors(h, hq, center, scale)
    limit = np.quantile(noise, .75)
    weak = hq >= limit
    p = np.random.default_rng(2026112602)
    shape = (2, len(held), held.shape[-1])
    probes = (p.normal(size=shape) + 1j * p.normal(size=shape)) / np.sqrt(2.)
    target = matched.filter_reflect(validation, probes)
    energy = np.mean(abs(target)**2, axis=-1).clip(1e-4)
    power = np.sum(abs(teacher)**2, axis=-1)
    emphasis = (1. + (noise >= limit)) / power.clip(.16)
    choices, matrices = [], []
    for index, weight in enumerate((np.ones(len(noise)), emphasis / emphasis.mean())):
        gram = design.T @ (design * weight[:, None])
        response = design.T @ (teacher * weight[:, None])
        trials = []
        for ridge in RIDGES:
            coeff = np.linalg.solve(gram + ridge * np.eye(len(gram)), response)
            predict = matched.filter_reflect(h @ coeff, probes)
            err = (np.mean(abs(predict - target)**2, axis=-1) / energy).mean(axis=0)
            values = [float(err.mean()), float(np.quantile(err, .95)),
                      float(err[weak].mean()), float(np.quantile(err[weak], .95))]
            objective = values[0] + .25 * values[1]
            if index == 1:
                objective = .25 * objective + values[2] + .25 * values[3]
            trials.append((objective, ridge, values, coeff))
        best = min(trials, key=lambda t: t[0])
        choices.append(dict(arm=ARMS[index], ridge=best[1], objective=best[0],
                            metrics="mean, P95, high-guard mean, high-guard P95", held_proxy=best[2]))
        matrices.append(best[3])
    matrices = np.stack(matrices)
    np.savez(output, center=center, scale=scale, matrices=matrices)
    write(output.parent / "configuration.json", dict(choices=choices, weak_guard=limit.item(),
          physical_fit=len(fit), physical_hold=len(hold), high_guard_views=int(weak.sum())))
    times = []
    for row in held[:48]:
        start = perf_counter()
        fit_kernel, fit_noise = model.estimate(row[None, :], prior, center, scale, matrices)
        times.append((perf_counter() - start) * 1000.)
        if not np.isfinite(fit_kernel).all() or np.any(fit_noise <= 0):
            raise ValueError("Nonfinite relative source smoke estimate")
    median, p95 = np.quantile(times, [.5, .95])
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(median), candidate_p95=float(p95), simultaneous_arms=len(ARMS)))
    print("R07_RELATIVE_PRIOR_FROZEN", [c['ridge'] for c in choices], flush=True)


def predict(received, prior_path, params, output):
    observations = np.load(received, allow_pickle=False)
    if observations.shape != (len(matched.SCENES), matched.COUNT, matched.model.receive.SAMPLES):
        raise ValueError("Expected only matched R07 received observations")
    prior = load_prior(prior_path)
    with np.load(params, allow_pickle=False) as a:
        cfg = {k: a[k] for k in ("center", "scale", "matrices")}
    kernel, noise = zip(*(model.estimate(row, prior, **cfg) for row in observations))
    np.savez(output, kernel=np.stack(kernel, axis=1), complex_variance=np.stack(noise))
    print("R07_RELATIVE_ESTIMATES_CLOSED", len(matched.SCENES), matched.COUNT, flush=True)


def score(root, reference):
    matched.score(root, reference, ARMS)


def dispatch(runtime):
    rt = read(runtime)
    root, native, prior, receive = (Path(rt[k]) for k in ("run_root", "reference", "prior", "source"))
    settings = root / "configuration.npz"
    prior_path = prior / "configuration.npz"
    steps = (("smoke", ()), ("tune", (receive, prior_path, settings)),
             ("predict", (prior / "evaluation_received.npy", prior_path, settings, root / "estimates.npz")),
             ("score", (root, native)))
    matched.protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(prior=str(runs / matched.RUN), reference=str(runs / matched.PARENT / "cache/validation_actions"),
                  source=str(runs / matched.protocol.RUN / "source_received.npy"))
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN, runner=__spec__.name,
              code_commit=transport.release_commit(repo),
              release_root=str(runs.parent / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              external_releases=list(inputs.values()), **inputs)
    transport.release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="action", required=True)
    pub = sub.add_parser("publish")
    for key in ("repo", "output", "context", "ssh-config"):
        pub.add_argument("--" + key, type=Path, required=True)
    steps = ((tune, ("source", "prior_path", "output")), (predict, ("received", "prior_path", "params", "output")),
             (score, ("root", "reference")), (dispatch, ("runtime",)))
    sub.add_parser("smoke")
    for f, fields in steps:
        stage = sub.add_parser(f.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    name = args.pop("action")
    {"publish": publish, "smoke": matched.smoke, **{f.__name__: f for f, _ in steps}}[name](**args)


if __name__ == "__main__":
    main()
