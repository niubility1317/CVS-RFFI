"""Blind Practical mixture with measured-noise gating and independent scoring."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
from scipy.special import expit

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_received as equal
from . import wisig_practical_receive_weighted as weighted
from . import wisig_practical_received_audit as protocol
from . import wisig_practical_receive_weighted_audit as comparison


RUN = "20261126-diagnostic-wisig-practical-received-gate-s2026112512-r01"
ARMS = ("receive_gate", "receive_gate_conservative_variance")
FRACTIONS = (.2, .4, .6, .8)
WIDTHS = (.20, .55, 1.)


def load_priors(first, second):
    with np.load(first, allow_pickle=False) as a:
        low = equal.Prior(**{name: a[name] for name in ("guard", "modes", "power", "log_noise")})
    with np.load(second, allow_pickle=False) as a:
        high = weighted.Prior(**{name: a[name] for name in ("guard", "color", "modes", "power")})
    return low, high


def gate(measured, center, width):
    mix = expit(np.log(center / measured) / width)
    return np.where(mix >= .99, 1., mix)


def equal_estimate(iq, prior, scale):
    projection = iq @ prior.modes.conj()
    variance = equal.noise_posterior(iq, projection, prior)
    strength = scale * prior.power
    gain = strength / (strength + variance[:, None])
    return (projection * gain) @ prior.modes.T, gain.mean(axis=-1)


def estimate(iq, first, second, cfg):
    projected_guard = abs(iq @ second.guard)**2
    measured = projected_guard.mean(axis=-1).clip(1e-12)
    mix = gate(measured, cfg["center"], cfg["width"])
    colored = weighted.guard_variance(projected_guard, second.color)
    conservative = weighted.guard_variance(projected_guard, second.color, 1. + 1.25 * (1. - mix))
    curve, _ = weighted.covariance_estimate(iq @ second.modes.conj(), second, cfg["weighted_scale"], colored)
    active = mix < 1.
    if np.any(active):
        broad, _ = equal_estimate(iq[active], first, cfg["equal_scale"])
        curve[active] = mix[active, None] * curve[active] + (1. - mix[active, None]) * broad
    return curve, colored, conservative


def tune(source, first, second, output):
    samples = np.load(source, allow_pickle=False)
    if samples.shape != (12, protocol.panel.NDEV, equal.SAMPLES):
        raise ValueError("Unexpected receive-only source matrix")
    a, b = load_priors(first, second)
    s0 = read(first.with_suffix(".json"))["scale"]
    s1 = read(second.with_suffix(".json"))["scales"][1]
    fit, hold = np.split(np.random.default_rng(protocol.panel.SEED).permutation(samples.shape[1]), [96])
    training = samples[:, fit].reshape(-1, equal.SAMPLES)
    held = samples[:, hold].reshape(-1, equal.SAMPLES).astype(np.complex128)
    power = abs(held @ b.guard)**2
    guard = power.mean(axis=-1).clip(1e-12)
    variance = weighted.guard_variance(power, b.color)
    base, dof_base = equal_estimate(held, a, s0)
    alternative, dof_alt = weighted.covariance_estimate(held @ b.modes.conj(), b, s1, variance)
    dof_alt = dof_alt.mean(axis=-1)
    signal = np.mean(abs(held)**2, axis=-1).clip(.1)
    ready = equal.guard_power(training, b.guard)
    candidates = []
    for fraction, center in zip(FRACTIONS, np.quantile(ready, FRACTIONS)):
        for width in WIDTHS:
            mix = gate(guard, center, width)
            guess = base * (1. - mix[:, None]) + alternative * mix[:, None]
            dof = dof_base * (1. - mix) + dof_alt * mix
            risk = (np.mean(abs(guess - held)**2, axis=-1) + guard * (2. * dof - 1.)) / signal
            mean, tail = float(risk.mean()), float(np.quantile(risk, .95))
            candidates.append(dict(fraction=fraction, center=float(center), width=width, mean=mean, p95=tail))
    settings = dict(min(candidates, key=lambda c: c["mean"] + .25 * c["p95"]), candidates=candidates,
                    equal_scale=s0, weighted_scale=s1, held_physical=len(hold))
    write(output, settings)
    times = []
    for packet in held[:48]:
        start = perf_counter()
        estimate(packet[None, :], a, b, settings)
        times.append((perf_counter() - start) * 1000)
    p50, p95 = np.quantile(times, [.5, .95])
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(p50), candidate_p95=float(p95)))
    print("PRACTICAL_RECEIVE_GATE_FROZEN", settings["center"], settings["width"], flush=True)


def predict(receive_path, first, second, settings, output):
    received = np.load(receive_path, allow_pickle=False)
    if received.shape != (*protocol.panel.SHAPE, equal.SAMPLES):
        raise ValueError("Wrong receive-only Practical matrix")
    a, b = load_priors(first, second)
    config = read(settings)
    waves, variances = [], []
    for row in received:
        waveform, raw, conservative = estimate(row, a, b, config)
        waves.append(waveform[:, protocol.panel.TEST])
        variances.append((raw, conservative))
    np.savez(output, wave=np.stack([waves, waves]), variance=np.stack(variances, axis=1))
    print("PRACTICAL_RECEIVE_GATE_ESTIMATES_CLOSED", *protocol.panel.SHAPE, flush=True)


def smoke():
    comparison.smoke()
    weights = gate(np.array([1e-4, .10, 1.]), .1, .55)
    if not (weights[0] == 1. and weights[1] == .5 and weights[2] < .1):
        raise ValueError("Observed-noise routing is reversed")
    print("PRACTICAL_RECEIVE_GATE_SMOKE_PASS", flush=True)


def score(root, reference):
    protocol.score(root, reference, arms=ARMS)


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt["run_root"])
    first, second = [Path(rt[name]) / "configuration.npz" for name in ("baseline", "weighted")]
    settings = root / "configuration.json"
    source = first.parent / "source_received.npy"
    received = first.parent / "evaluation_received.npy"
    phases = (("smoke", ()), ("tune", (source, first, second, settings)),
              ("predict", (received, first, second, settings, root / "estimates.npz")),
              ("score", (root, rt["reference"])))
    protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, phases)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(baseline=str(runs / protocol.RUN), weighted=str(runs / comparison.RUN),
                  reference=str(runs / protocol.panel.full.RUN / "reference.npz"))
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
    steps = ((smoke, ()), (tune, ("source", "first", "second", "output")),
             (predict, ("receive_path", "first", "second", "settings", "output")),
             (score, ("root", "reference")), (dispatch, ("runtime",)))
    for func, fields in steps:
        stage = sub.add_parser(func.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    name = args.pop("action")
    {"publish": publish, **{f.__name__: f for f, _ in steps}}[name](**args)


if __name__ == "__main__":
    main()
