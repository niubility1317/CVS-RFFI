"""Fixed Practical source-V diagnostic of weighted receive-only estimates."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_receive_weighted as model
from . import wisig_practical_received as previous
from . import wisig_practical_received_audit as protocol


RUN = "20261126-diagnostic-wisig-practical-received-weighted-s2026112512-r01"
ARMS = ("weighted_guard", "weighted_colored_guard", "weighted_with_repeat")
SCALES = (.3, .6, 1., 2., 4., 8.)
PENALTY = 1.


def smoke():
    protocol.smoke()
    rng = np.random.default_rng(607)
    guard = previous.guard_basis()
    color = np.r_[np.zeros(guard.shape[1] - 6), np.full(6, guard.shape[1] / 6)]
    v = .03 + .04 * color
    tones = (rng.normal(size=(240, len(v))) + 1j * rng.normal(size=(240, len(v)))) * np.sqrt(v / 2)
    samples = tones @ guard.T
    fitted = model.guard_variance(abs(samples @ guard)**2, color, PENALTY)
    if abs(np.median(fitted) - .03) > .010:
        raise ValueError("Colored guard did not recover the synthetic white component")
    print("PRACTICAL_WEIGHTED_SMOKE_PASS", flush=True)


def estimate(iq, prior, settings):
    iq = np.asarray(iq, dtype=np.complex128)
    power = abs(iq @ prior.guard)**2
    guard = power.mean(axis=-1).clip(1e-12)
    colored = model.guard_variance(power, prior.color, PENALTY)
    observed = iq @ prior.modes.conj()
    raw, _ = model.covariance_estimate(observed, prior, settings["scales"][0], guard)
    light, _ = model.covariance_estimate(observed, prior, settings["scales"][1], colored)
    repeat = model.repeat_refinement(iq, light, colored, .65)
    return (raw, light, repeat), (guard, colored, colored)


def tune(source, output):
    samples = np.load(source, allow_pickle=False)
    if samples.shape != (12, protocol.panel.NDEV, previous.SAMPLES):
        raise ValueError("Unexpected receive-only source matrix")
    index = np.random.default_rng(protocol.panel.SEED).permutation(samples.shape[1])
    fit, hold = np.split(index, [96])
    train = samples[:, fit].reshape(-1, previous.SAMPLES)
    held = samples[:, hold].reshape(-1, previous.SAMPLES).astype(np.complex128)
    prior = model.fit(train, exponent=.5, penalty=PENALTY)
    power = abs(held @ prior.guard)**2
    guard = power.mean(axis=-1).clip(1e-12)
    colored = model.guard_variance(power, prior.color, PENALTY)
    observed = held @ prior.modes.conj()
    estimated_signal = np.mean(abs(held)**2, axis=-1).clip(.10)
    results = []
    for variance in (guard, colored):
        objective = []
        for scale in SCALES:
            wave, gain = model.covariance_estimate(observed, prior, scale, variance)
            error = np.mean(abs(wave - held)**2, axis=-1)
            sure = (error + guard * (2. * gain.mean(axis=-1) - 1.)) / estimated_signal
            objective.append(float(sure.mean() + .25 * np.quantile(sure, .95)))
        results.append(objective)
    settings = dict(scales=[SCALES[int(np.argmin(v))] for v in results], objective=results,
                    physical_fit=len(fit), physical_hold=len(hold), variance_penalty=PENALTY)
    np.savez(output, guard=prior.guard, color=prior.color, modes=prior.modes, power=prior.power)
    write(output.with_suffix(".json"), settings)
    times = []
    for packet in held[:48]:
        start = perf_counter()
        estimate(packet[None, :], prior, settings)
        times.append(1000 * (perf_counter() - start))
    p50, p95 = np.quantile(times, [.50, .95])
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(p50), candidate_p95=float(p95), simultaneous_arms=len(ARMS)))
    print("PRACTICAL_WEIGHTED_FROZEN", settings["scales"], flush=True)


def predict(receive_path, params, output):
    samples = np.load(receive_path, allow_pickle=False)
    if samples.shape != (*protocol.panel.SHAPE, previous.SAMPLES):
        raise ValueError("Wrong receive-only Practical matrix")
    with np.load(params, allow_pickle=False) as record:
        prior = model.Prior(**{key: record[key] for key in ("guard", "color", "modes", "power")})
    settings = read(params.with_suffix(".json"))
    waves, variances = [], []
    for row in samples:
        wave, variance = estimate(row, prior, settings)
        waves.append(np.stack(wave)[..., protocol.panel.TEST])
        variances.append(np.stack(variance))
    np.savez(output, wave=np.stack(waves, axis=1), variance=np.stack(variances, axis=1))
    print("PRACTICAL_WEIGHTED_ESTIMATES_CLOSED", *protocol.panel.SHAPE, flush=True)


def score(root, reference):
    protocol.score(root, reference, arms=ARMS)


def dispatch(runtime):
    rt = read(runtime)
    root, ref, baseline = map(Path, (rt["run_root"], rt["reference"], rt["baseline"]))
    config = root / "configuration.npz"
    phases = (("smoke", ()), ("tune", (baseline / "source_received.npy", config)),
              ("predict", (baseline / "evaluation_received.npy", config, root / "estimates.npz")),
              ("score", (root, ref)))
    protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, phases)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    paths = dict(baseline=str(runs / protocol.RUN), reference=str(runs / protocol.panel.full.RUN / "reference.npz"))
    rt = dict(project=ctx["project"], python=ctx["python"], frozen_sources={}, run_id=RUN, runner=__spec__.name,
              code_commit=transport.release_commit(repo),
              release_root=str(runs.parent / "releases" / f"{RUN}-release"), run_root=str(runs / RUN),
              external_releases=list(paths.values()), **paths)
    transport.release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="action", required=True)
    pub = sub.add_parser("publish")
    for arg in ("repo", "output", "context", "ssh-config"):
        pub.add_argument("--" + arg, type=Path, required=True)
    steps = ((smoke, ()), (tune, ("source", "output")), (predict, ("receive_path", "params", "output")),
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
