"""Test the receive-only coherence gate against the fixed R07 probes."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_audit as matched
from . import wisig_practical_probe_gate as model
from . import wisig_practical_probe_relative_audit as relative


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r03"
ARMS = ("coherence_gate", "coherence_gate_colored", "coherence_gate_pooled", "conservative_gate_pooled")


def load_operators(first, second):
    with np.load(second, allow_pickle=False) as cfg:
        return relative.load_prior(first), cfg["center"], cfg["scale"], cfg["matrices"][0]


def smoke():
    matched.smoke()
    guard = model.blind.receive.guard_basis()
    span = model.repeat_basis(guard)
    np.testing.assert_allclose(guard.T @ span, 0., atol=3e-10)
    np.testing.assert_allclose(span.T @ span, np.eye(span.shape[1]), atol=3e-10)
    rng = np.random.default_rng(4622)
    z = (rng.normal(size=(1024, len(guard))) + 1j * rng.normal(size=(1024, len(guard)))) * .1
    configured = model.NoisePrior(guard, np.ones(guard.shape[-1]), span[:, :model.REPEAT_MODES])
    for power in model.variance_candidates(z, configured):
        if not .019 < power.mean() < .021:
            raise ValueError("Orthogonal noise component smoke failed")
    print("R07_PROBE_COHERENCE_GATE_SMOKE_PASS", flush=True)


def fit(source, first, second, output):
    bank = np.load(source, allow_pickle=False)
    if bank.shape != (12, matched.protocol.panel.NDEV, model.blind.receive.SAMPLES):
        raise ValueError("The receive-only source U bank is required")
    operators = load_operators(first, second)
    fit_rows, held_rows = np.split(np.random.default_rng(matched.protocol.panel.SEED).permutation(bank.shape[1]), [96])
    train, held = (bank[:, ids].reshape(-1, bank.shape[-1]).astype(np.complex128) for ids in (fit_rows, held_rows))
    prior = model.fit_noise(train)
    np.savez(output, **vars(prior))
    write(output.parent / "configuration.json", dict(physical_fit=len(fit_rows), physical_hold=len(held_rows),
          guard_dimensions=prior.guard.shape[1], orthogonal_repeat_dimensions=prior.null.shape[1],
          diffuse_rule="one minus frozen source phase concentration", conservative_exponent=1.5))
    times = []
    for row in held[:48]:
        start = perf_counter()
        action, variance = model.estimate(row[None, :], *operators, prior)
        times.append((perf_counter() - start) * 1000)
        if not np.isfinite(action).all() or not np.isfinite(variance).all() or np.any(variance <= 0.):
            raise ValueError("Nonfinite receive-only source smoke estimate")
    p50, p95 = np.quantile(times, [.5, .95])
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(p50), candidate_p95=float(p95), simultaneous_arms=len(ARMS)))
    print("R07_PROBE_COHERENCE_GATE_FROZEN", len(fit_rows), prior.null.shape[1], flush=True)


def predict(received, first, second, params, output):
    observations = np.load(received, allow_pickle=False)
    if observations.shape != (len(matched.SCENES), matched.COUNT, model.blind.receive.SAMPLES):
        raise ValueError("Expected only matched R07 received observations")
    operators = load_operators(first, second)
    with np.load(params, allow_pickle=False) as cfg:
        prior = model.NoisePrior(cfg["guard"], cfg["color"], cfg["null"])
    kernel, noise = zip(*(model.estimate(row, *operators, prior) for row in observations))
    np.savez(output, kernel=np.stack(kernel, axis=1), complex_variance=np.stack(noise, axis=1))
    print("R07_PROBE_COHERENCE_GATE_CLOSED", len(matched.SCENES), matched.COUNT, flush=True)


def score(root, reference):
    matched.score(root, reference, ARMS)


def dispatch(runtime):
    rt = read(runtime)
    root, first, second, source, reference = (Path(rt[key]) for key in ("run_root", "first", "second", "source", "reference"))
    params = root / "configuration.npz"
    p1, p2 = first / "configuration.npz", second / "configuration.npz"
    steps = (("smoke", ()), ("fit", (source, p1, p2, params)),
             ("predict", (first / "evaluation_received.npy", p1, p2, params, root / "estimates.npz")),
             ("score", (root, reference)))
    matched.protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(first=str(runs / matched.RUN), second=str(runs / relative.RUN),
                  source=str(runs / matched.protocol.RUN / "source_received.npy"),
                  reference=str(runs / matched.PARENT / "cache/validation_actions"))
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
    steps = ((smoke, ()), (fit, ("source", "first", "second", "output")),
             (predict, ("received", "first", "second", "params", "output")),
             (score, ("root", "reference")), (dispatch, ("runtime",)))
    for function, fields in steps:
        stage = sub.add_parser(function.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    name = args.pop("action")
    {"publish": publish, **{f.__name__: f for f, _ in steps}}[name](**args)


if __name__ == "__main__":
    main()
