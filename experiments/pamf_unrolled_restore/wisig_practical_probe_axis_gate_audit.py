"""Evaluate cross-fitted path weights on the unchanged R07 reference."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_axis_gate as model
from . import wisig_practical_probe_scale_audit as scale


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r09"
ARMS = ("frozen_soft_sparse", "crossfit_path_weights")
sparse = scale.sparse
matched = scale.matched


def estimator(first, third, source, protection):
    prior = sparse.previous.gate_audit.relative.load_prior(first)
    with np.load(source, allow_pickle=False) as archive:
        return sparse.previous.fused.prepare(prior, archive["center"], archive["scale"], archive["matrices"][0],
                                             sparse.previous.noise_prior(third), protection)


def estimate(received, cfg, knots, width, coefficients, strength):
    design, direct, diffuse, (noise, _, _, variance) = sparse.previous.fused.components(received, cfg)
    broad = design @ cfg.matrix
    base = direct + diffuse[:, None] * (broad - direct)
    features = model.gate_features(noise, broad, cfg.prior.anchor, knots, width)
    adapted = model.blend(direct, broad, diffuse, cfg.prior.anchor, features, coefficients, strength)
    return np.stack((base, adapted)), variance


def tune(source, first, third, frozen, output):
    received = np.load(source, allow_pickle=False)
    if received.shape != (len(matched.SCENES), scale.EXPANDED_FIT + scale.HOLD, matched.model.receive.SAMPLES):
        raise ValueError("Expected the frozen Practical 4096/192 receive-only source bank")
    base_settings = read(frozen.with_suffix(".json"))
    protection = base_settings["protection_scale"]
    old_ridge = base_settings["choices"][0]["ridge"]
    cfg = estimator(first, third, frozen, protection)
    source_fit, source_hold = (z.transpose(1, 0, 2).astype(np.complex128) for z in
                               (received[:, :scale.EXPANDED_FIT], received[:, scale.EXPANDED_FIT:]))
    train, held = (x.reshape(-1, received.shape[-1]) for x in (source_fit, source_hold))
    teacher = sparse.sparse.target_columns(source_fit, cfg.prior)
    design, direct, diffuse, (noise, _, _, _) = sparse.previous.fused.components(train, cfg)
    val_design, val_direct, val_diffuse, (quality, _, _, _) = sparse.previous.fused.components(held, cfg)
    old = val_design @ cfg.matrix
    folds = np.random.default_rng(693).permutation(len(source_fit)) % 4
    fold = np.repeat(folds, len(matched.SCENES))
    crossfit = np.empty_like(teacher)
    penalty = old_ridge * np.eye(design.shape[-1])
    for part in range(4):
        use = fold != part
        x = design[use]
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ teacher[use])
        crossfit[~use] = design[~use] @ weights
    hold_targets = [f(source_hold, cfg.prior) for f in (sparse.sparse.target_columns, sparse.sparse.relative.target_columns)]
    rng = np.random.default_rng(2026112602)
    shape = (2, *held.shape)
    probes = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.)
    refs = [matched.filter_reflect(h, probes) for h in hold_targets]
    weak = quality >= np.quantile(noise, .75)

    def metrics(h):
        actual = matched.filter_reflect(h, probes)
        error = np.mean([(np.mean(abs(actual-y)**2, axis=-1) / np.mean(abs(y)**2, axis=-1).clip(1e-4)).mean(0)
                         for y in refs], axis=0)
        w = error[weak]
        return [float(z) for z in (error.mean(), w.mean(), np.quantile(w, .9), np.quantile(w, .95))]

    baseline = metrics(val_direct + val_diffuse[:, None] * (old - val_direct))
    knots = np.quantile(np.log(noise), [.12, .35, .58, .78, .92])
    trials = []
    for width in (.55, 1.):
        train_features = model.gate_features(noise, crossfit, cfg.prior.anchor, knots, width)
        held_features = model.gate_features(quality, old, cfg.prior.anchor, knots, width)
        for ridge in (.1, 1., 10.):
            weights = model.solve_gate(train_features, direct, crossfit, diffuse, teacher, cfg.prior.anchor, ridge)
            for strength in (.5, 1.):
                y = model.blend(val_direct, old, val_diffuse, cfg.prior.anchor, held_features, weights, strength)
                values = metrics(y)
                objective = np.dot(values, [1., .55, .22, .20])
                trials.append((float(objective), values, width, ridge, strength, weights))
    objective, values, width, ridge, strength, weights = min(trials, key=lambda z: z[0])
    np.savez(output, knots=knots, coefficients=weights)
    write(output.with_suffix(".json"), dict(width=width, ridge=ridge, strength=strength, protection_scale=protection,
          physical_fit=len(source_fit), physical_hold=len(source_hold), baseline=baseline, selected=values,
          objective=objective, proxy_columns=["mean", "high_guard_mean", "high_guard_p90", "high_guard_p95"]))
    times = []
    for row in held[:48, None, :]:
        start = perf_counter()
        kernel, variance = estimate(row, cfg, knots, width, weights, strength)
        times.append((perf_counter() - start) * 1000.)
        if not np.isfinite(kernel).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError("Invalid source smoke output")
    p50, p95 = np.quantile(times, [.5, .95])
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(p50), candidate_p95=float(p95), simultaneous_arms=len(ARMS)))
    print("R07_PATH_WEIGHTS_FROZEN", width, ridge, strength, flush=True)


def predict(received, first, third, frozen, params, output):
    data = np.load(received, allow_pickle=False)
    if data.shape != (len(matched.SCENES), matched.COUNT, matched.model.receive.SAMPLES):
        raise ValueError("Expected only the matched R07 received observations")
    settings = read(params.with_suffix(".json"))
    cfg = estimator(first, third, frozen, settings["protection_scale"])
    with np.load(params, allow_pickle=False) as f:
        args = (cfg, f["knots"], settings["width"], f["coefficients"], settings["strength"])
    kernels, noise = zip(*(estimate(row, *args) for row in data))
    np.savez(output, kernel=np.stack(kernels, axis=1), complex_variance=np.stack(noise))
    print("R07_PATH_WEIGHTS_ESTIMATES_CLOSED", len(matched.SCENES), matched.COUNT, flush=True)


def score(root, reference):
    matched.score(root, reference, ARMS)


def dispatch(runtime):
    rt = read(runtime)
    root, first, third, frozen, reference = (Path(rt[k]) for k in ("run_root", "first", "third", "frozen", "reference"))
    p1, p3, baseline = (p / "configuration.npz" for p in (first, third, frozen))
    opts = root / "configuration.npz"
    steps = (("smoke", ()), ("tune", (frozen / "source_received.npy", p1, p3, baseline, opts)),
             ("predict", (first / "evaluation_received.npy", p1, p3, baseline, opts, root / "estimates.npz")),
             ("score", (root, reference)))
    matched.protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    paths = dict(first=str(runs / matched.RUN), third=str(runs / sparse.previous.gate_audit.RUN),
                 frozen=str(runs / scale.EXPANDED_RUN), reference=str(runs / matched.PARENT / "cache/validation_actions"))
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
    steps = ((tune, ("source", "first", "third", "frozen", "output")),
             (predict, ("received", "first", "third", "frozen", "params", "output")),
             (score, ("root", "reference")), (dispatch, ("runtime",)))
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
