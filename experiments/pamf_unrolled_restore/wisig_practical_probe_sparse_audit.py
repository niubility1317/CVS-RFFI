"""Score receive-only sparse training against the same archived R07 probes."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_fused_audit as previous
from . import wisig_practical_probe_sparse as sparse


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r05"
ARMS = ("soft_sparse_gate", "sparse_dense_gate")
RIDGES = (40., 120., 320., 800.)
matched = previous.matched


def estimate(received, cfg):
    design, direct, diffuse, (_, _, _, variance) = previous.fused.components(received, cfg)
    regression = design[None, ...] @ cfg.matrix
    return direct + diffuse[:, None] * (regression - direct), variance


def tune(source, first, second, third, fourth, output):
    bank = np.load(source, allow_pickle=False)
    if bank.shape != (12, matched.protocol.panel.NDEV, matched.model.receive.SAMPLES):
        raise ValueError("Expected the original receive-only source U bank")
    fit, held = np.split(np.random.default_rng(matched.protocol.panel.SEED).permutation(bank.shape[1]), [96])
    fit_bank(bank, fit, held, first, second, third, fourth, output)


def fit_bank(bank, fit, held, first, second, third, fourth, output):
    operators = previous.gate_audit.load_operators(first, second)
    noise_prior = previous.noise_prior(third)
    protection = read(fourth)["protection_scale"]
    prior, _, _, original = operators
    source_rows, held_rows = (bank[:, ids].transpose(1, 0, 2).astype(np.complex128) for ids in (fit, held))
    dense, compact = (f(source_rows, prior) for f in (sparse.relative.target_columns, sparse.target_columns))
    held_dense, held_sparse = (f(held_rows, prior) for f in (sparse.relative.target_columns, sparse.target_columns))
    source_iq = source_rows.reshape(-1, source_rows.shape[-1])
    held_iq = held_rows.reshape(-1, held_rows.shape[-1])
    features, q = sparse.relative.observables(source_iq, prior)
    center, scale = features.mean(axis=0), features.std(axis=0).clip(.05)
    design = sparse.relative.regressors(features, q, center, scale)
    gram = design.T @ design
    initial = previous.fused.prepare(*operators, noise_prior, protection)
    setup = previous.fused.prepare(prior, center, scale, original, noise_prior, protection)
    unseen, direct, diffuse, (noise, _, _, _) = previous.fused.components(held_iq, setup)
    high = noise > np.quantile(q, .75)
    rng = np.random.default_rng(2026112602)
    shape = (2, *held_iq.shape)
    probes = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.)
    expected = [matched.filter_reflect(target, probes) for target in (held_dense, held_sparse)]

    def metric(kernels):
        prediction = matched.filter_reflect(kernels, probes)
        error = np.mean([
            (np.mean(abs(prediction - target)**2, axis=-1) / np.mean(abs(target)**2, axis=-1).clip(1e-4)).mean(axis=0)
            for target in expected], axis=0)
        return [float(error.mean()), float(np.quantile(error, .95)),
                float(error[high].mean()), float(np.quantile(error[high], .95))]

    base = metric(previous.fused.estimate(held_iq, initial)[0][0])
    matrices, choices = [], []
    for arm, target in zip(ARMS, (compact, .65 * compact + .35 * dense)):
        rhs = design.T @ target
        trials = []
        for ridge in RIDGES:
            matrix = np.linalg.solve(gram + ridge * np.eye(len(gram)), rhs)
            kernel = direct + diffuse[:, None] * (unseen @ matrix - direct)
            value = metric(kernel)
            objective = value[0] + .25 * value[1] + .55 * value[2] + .2 * value[3]
            trials.append((objective, ridge, value, matrix))
        objective, ridge, value, matrix = min(trials, key=lambda x: x[0])
        matrices.append(matrix)
        choices.append(dict(arm=arm, ridge=ridge, objective=objective, held_operator_proxy=value))
    matrices = np.stack(matrices)
    np.savez(output, center=center, scale=scale, matrices=matrices)
    write(output.parent / "configuration.json", dict(physical_fit=len(fit), physical_hold=len(held), choices=choices,
          baseline_held_operator_proxy=base, protection_scale=protection))
    cfg = previous.fused.prepare(prior, center, scale, matrices, noise_prior, protection)
    times = []
    for iq in held_iq[:48, None, :]:
        start = perf_counter()
        kernel, variance = estimate(iq, cfg)
        times.append((perf_counter() - start) * 1000.)
        if not np.isfinite(kernel).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError("Invalid receive-only source smoke output")
    p50, p95 = np.quantile(times, [.5, .95])
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(p50), candidate_p95=float(p95), simultaneous_arms=len(ARMS)))
    print("R07_SOFT_SPARSE_PRIOR_FROZEN", [m["ridge"] for m in choices], flush=True)


def predict(received, first, third, params, output):
    data = np.load(received, allow_pickle=False)
    if data.shape != (len(matched.SCENES), matched.COUNT, matched.model.receive.SAMPLES):
        raise ValueError("Expected only the matched R07 received observations")
    prior = previous.gate_audit.relative.load_prior(first)
    with np.load(params, allow_pickle=False) as f:
        cfg = previous.fused.prepare(prior, f["center"], f["scale"], f["matrices"], previous.noise_prior(third),
                                     read(params.with_suffix(".json"))["protection_scale"])
    kernels, variance = zip(*(estimate(row, cfg) for row in data))
    np.savez(output, kernel=np.stack(kernels, axis=1), complex_variance=np.stack(variance))
    print("R07_SOFT_SPARSE_ESTIMATES_CLOSED", len(matched.SCENES), matched.COUNT, flush=True)


def score(root, reference):
    matched.score(root, reference, ARMS)


def dispatch(runtime):
    rt = read(runtime)
    root, first, second, third, fourth, source, reference = [Path(rt[k]) for k in (
        "run_root", "first", "second", "third", "fourth", "source", "reference")]
    p1, p2, p3 = (p / "configuration.npz" for p in (first, second, third))
    opts = root / "configuration.npz"
    steps = (("smoke", ()), ("tune", (source, p1, p2, p3, fourth / "configuration.json", opts)),
             ("predict", (first / "evaluation_received.npy", p1, p3, opts, root / "estimates.npz")),
             ("score", (root, reference)))
    matched.protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    paths = dict(first=str(runs / matched.RUN), second=str(runs / previous.gate_audit.relative.RUN),
                 third=str(runs / previous.gate_audit.RUN), fourth=str(runs / previous.RUN),
                 source=str(runs / matched.protocol.RUN / "source_received.npy"),
                 reference=str(runs / matched.PARENT / "cache/validation_actions"))
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
    steps = ((tune, ("source", "first", "second", "third", "fourth", "output")),
             (predict, ("received", "first", "third", "params", "output")),
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
