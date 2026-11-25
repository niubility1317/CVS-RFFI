"""Same-reference scoring and paired CPU timings for the shared pilot implementation."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_fused as fused
from . import wisig_practical_probe_gate_audit as gate_audit


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r04"
ARMS = (*gate_audit.ARMS, "coherence_gate_protected_noise")
matched = gate_audit.matched


def noise_prior(path):
    with np.load(path, allow_pickle=False) as a:
        return fused.gate.NoisePrior(a["guard"], a["color"], a["null"])


def benchmark(source, first, second, noise_path, output):
    bank = np.load(source, allow_pickle=False)
    if bank.shape != (12, matched.protocol.panel.NDEV, fused.gate.blind.receive.SAMPLES):
        raise ValueError("The receive-only source U bank is required")
    fit, held = np.split(np.random.default_rng(matched.protocol.panel.SEED).permutation(bank.shape[1]), [96])
    priors = gate_audit.load_operators(first, second)
    n = noise_prior(noise_path)
    train = bank[:, fit].reshape(-1, bank.shape[-1])
    power = fused.gate.blind.receive.guard_power(train, priors[0].guard)
    scale = float(np.quantile(power, .75))
    write(output, dict(physical_fit=len(fit), physical_hold=len(held), protection_scale=scale))
    cfg = fused.prepare(*priors, n, scale)
    probe = bank[:, held].reshape(-1, bank.shape[-1])[:48].astype(np.complex128)
    expected = fused.gate.estimate(probe, *priors, n)
    adapted = fused.estimate(probe, cfg)
    for actual, original in zip(adapted, expected):
        np.testing.assert_allclose(np.stack(actual)[:len(gate_audit.ARMS)], original, atol=1e-12, rtol=1e-9)
    durations = [[], []]
    methods = ((fused.gate.estimate, (*priors, n)), (fused.estimate, (cfg,)))
    for y in probe[:, None, :]:
        measured = [[], []]
        for repeat in range(6):
            for method in (repeat % 2, 1 - repeat % 2):
                function, args = methods[method]
                start = perf_counter()
                function(y, *args)
                measured[method].append((perf_counter() - start) * 1000.)
        for result, ts in zip(durations, measured):
            result.append(np.median(ts))
    q, v = np.quantile(durations, [.5, .95], axis=1)
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(probe),
          paired_same_process=True, baseline_median=float(q[0]), baseline_p95=float(v[0]),
          candidate_median=float(q[1]), candidate_p95=float(v[1]), simultaneous_arms=len(ARMS)))
    print("R07_FUSED_PILOT_PRIOR_FROZEN", scale, flush=True)


def predict(received, first, second, noise_path, config, output):
    data = np.load(received, allow_pickle=False)
    if data.shape != (len(matched.SCENES), matched.COUNT, fused.gate.blind.receive.SAMPLES):
        raise ValueError("Expected only matched R07 received observations")
    cfg = fused.prepare(*gate_audit.load_operators(first, second), noise_prior(noise_path), read(config)["protection_scale"])
    kernel, variance = zip(*(fused.estimate(row, cfg) for row in data))
    np.savez(output, kernel=np.stack(kernel, axis=1), complex_variance=np.stack(variance, axis=1))
    print("R07_FUSED_PILOT_CLOSED", len(matched.SCENES), matched.COUNT, flush=True)


def score(root, reference):
    matched.score(root, reference, ARMS)


def dispatch(runtime):
    rt = read(runtime)
    root, first, second, third, source, reference = [Path(rt[k]) for k in ("run_root", "first", "second", "third", "source", "reference")]
    p1, p2, p3 = (p / "configuration.npz" for p in (first, second, third))
    config = root / "configuration.json"
    steps = (("smoke", ()), ("benchmark", (source, p1, p2, p3, config)),
             ("predict", (first / "evaluation_received.npy", p1, p2, p3, config, root / "estimates.npz")),
             ("score", (root, reference)))
    matched.protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(first=str(runs / matched.RUN), second=str(runs / gate_audit.relative.RUN), third=str(runs / gate_audit.RUN),
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
    steps = ((benchmark, ("source", "first", "second", "noise_path", "output")),
             (predict, ("received", "first", "second", "noise_path", "config", "output")),
             (score, ("root", "reference")), (dispatch, ("runtime",)))
    sub.add_parser("smoke")
    for f, fields in steps:
        stage = sub.add_parser(f.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    name = args.pop("action")
    {"publish": publish, "smoke": gate_audit.smoke, **{f.__name__: f for f, _ in steps}}[name](**args)


if __name__ == "__main__":
    main()
