"""Freeze and score the Practical receive-only diagnostic in separate processes."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_fast_audit as panel
from . import wisig_practical_received as model


RUN = "20261126-diagnostic-wisig-practical-received-s2026112512-r01"
ARMS = ("raw_and_guard", "raw_and_repeat", "receive_covariance")
SCALES = (.3, .6, 1., 2., 4., 8.)


def smoke():
    rng = np.random.default_rng(947)
    tones = np.arange(1, 13) / 26
    tones = tones[tones < .32]
    basis = np.exp(2j * np.pi * model.SAMPLES**-1 * np.outer(np.arange(model.SAMPLES),
                                                            np.rint(tones * model.SAMPLES)))
    signal = (rng.normal(size=(400, len(tones))) + 1j * rng.normal(size=(400, len(tones)))) @ basis.T
    noisy = signal + .1 * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    prior = model.fit(noisy[:320])
    waveform, noise = model.estimates(noisy[320:], prior)
    if not np.isfinite(waveform).all() or np.any(noise <= 0):
        raise ValueError("Nonfinite blind estimate")
    if np.mean(abs(waveform[0] - signal[320:])**2) >= .35 * np.mean(abs(noisy[320:] - signal[320:])**2):
        raise ValueError("Receive covariance did not recover the public-tone signal")
    print("PRACTICAL_RECEIVE_ONLY_SMOKE_PASS", flush=True)


def prepare(development, challenge, root):
    # The export process sees the old archive handles. It reads only `received`.
    with np.load(development, allow_pickle=False) as stored:
        source_received = stored["received"]
    with np.load(challenge, allow_pickle=False) as stored:
        evaluation_received = stored["received"][list(panel.SLOTS)]
    np.save(root / "source_received.npy", source_received, allow_pickle=False)
    np.save(root / "evaluation_received.npy", evaluation_received, allow_pickle=False)


def tune(source, output):
    received = np.load(source, allow_pickle=False)
    if received.shape != (12, panel.NDEV, model.SAMPLES):
        raise ValueError("Unexpected receive-only source matrix")
    fit, hold = np.split(np.random.default_rng(panel.SEED).permutation(panel.NDEV), [96])
    prior = model.fit(received[:, fit].reshape(-1, model.SAMPLES))
    held = received[:, hold].reshape(-1, model.SAMPLES)
    waves, variance = model.estimates(held, prior, SCALES)
    strengths = np.asarray(SCALES)[:, None, None] * prior.power
    gains = strengths / (strengths + variance[None, :, None])
    # Conditional Stein risk surrogate with a receive-only noise estimate.
    correction = variance[None, :] * (2. * gains.mean(axis=-1) - 1.)
    risk = np.mean(np.abs(waves - held[None, ...])**2, axis=-1) + correction
    scale = np.mean(np.abs(held)**2, axis=-1).clip(1e-12)
    objective = (risk / scale).mean(axis=-1)
    best = int(np.argmin(objective))
    np.savez(output, guard=prior.guard, modes=prior.modes, power=prior.power, log_noise=prior.log_noise)
    write(output.with_suffix(".json"), dict(scale=SCALES[best], source_risk=objective.tolist(),
                                            physical_fit=len(fit), physical_hold=len(hold)))
    times = []
    for packet in held[:48]:
        start = perf_counter()
        model.estimates(packet[None, :], prior, (SCALES[best],))
        times.append(1000. * (perf_counter() - start))
    write(output.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(np.median(times)), candidate_p95=float(np.quantile(times, .95))))
    print("PRACTICAL_RECEIVE_ONLY_FROZEN", SCALES[best], flush=True)


def predict(receive_path, params, output):
    from .wisig_short_training import _estimate

    received = np.load(receive_path, allow_pickle=False)
    if received.shape != (*panel.SHAPE, model.SAMPLES):
        raise ValueError("Wrong receive-only evaluation matrix")
    with np.load(params, allow_pickle=False) as stored:
        prior = model.Prior(**{name: stored[name] for name in ("guard", "modes", "power", "log_noise")})
    scale = read(params.with_suffix(".json"))["scale"]
    waves, variance = [], []
    for row in received:
        waveform, sigma = model.estimates(row, prior, (scale,))
        guard = model.guard_power(row, prior.guard)
        repeat = _estimate(row)[0].effective_variance.clip(1e-12)
        waves.append(np.stack((row[:, panel.TEST], row[:, panel.TEST], waveform[0][:, panel.TEST])))
        variance.append(np.stack((guard, repeat, sigma)))
    np.savez(output, wave=np.stack(waves, axis=1), variance=np.stack(variance, axis=1))
    print("PRACTICAL_RECEIVE_ONLY_ESTIMATES_CLOSED", *panel.SHAPE, flush=True)


def score(root, reference, arms=ARMS):
    with np.load(root / "estimates.npz", allow_pickle=False) as stored:
        waves, variance = stored["wave"], stored["variance"]
    if waves.shape != (len(arms), *panel.SHAPE, len(panel.TEST)) or variance.shape != (len(arms), *panel.SHAPE):
        raise ValueError("Incomplete receive-only estimates")
    if not np.isfinite(waves).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Invalid receive-only estimates")
    with np.load(reference, allow_pickle=False) as stored:
        signal = stored["signal"][list(panel.SLOTS)][:, :, panel.TEST]
        true_variance = stored["variance"][list(panel.SLOTS)]
    power = np.mean(abs(signal)**2, axis=-1).clip(1e-18)
    wave_error = np.mean(abs(waves - signal)**2, axis=-1) / power
    noise_error = abs(10. * np.log10(variance / true_variance))

    def stats(wave, noise):
        return dict(records=wave.size, mean=float(wave.mean()), median=float(np.median(wave)),
                    p90=float(np.quantile(wave, .90)), p95=float(np.quantile(wave, .95)),
                    noise_median=float(np.median(noise)), noise_p95=float(np.quantile(noise, .95)))

    labels = panel.LABELS
    primary = [i for i, label in enumerate(labels) if not label.endswith("/double_power")]
    groups = dict(primary=primary, low_urban=[i for i in primary if "practical_low_urban" in labels[i]],
                  doubled=list(range(len(primary), len(labels))))
    results = {}
    for name, wave, noise in zip(arms, wave_error, noise_error):
        results[name] = {key: stats(wave[idx], noise[idx]) for key, idx in groups.items()}
        results[name]["rows"] = {label: stats(w, v) for label, w, v in zip(labels, wave, noise)}
    write(root / "scores.json", dict(status="ANALYZED", truth_side_last=True, receive_only=True,
          settings=read(root / "configuration.json"), speed=read(root / "speed.json"), results=results))
    print("PRACTICAL_RECEIVE_ONLY_SCORED", len(labels), flush=True)


def dispatch(runtime):
    rt = read(runtime)
    root, parent, prior = Path(rt["run_root"]), Path(rt["parent"]), Path(rt["previous"])
    config = root / "configuration.npz"
    phases = (("smoke", ()), ("prepare", (prior / "development.npz", parent / "paired.npz", root)),
              ("tune", (root / "source_received.npy", config)),
              ("predict", (root / "evaluation_received.npy", config, root / "estimates.npz")),
              ("score", (root, parent / "reference.npz")))
    panel.full.source.parent.parent.prior_runner.dispatch(rt, phases)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(parent=str(runs / panel.full.RUN), previous=str(runs / panel.RUN))
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
    steps = ((smoke, ()), (prepare, ("development", "challenge", "root")), (tune, ("source", "output")),
             (predict, ("receive_path", "params", "output")), (score, ("root", "reference")),
             (dispatch, ("runtime",)))
    for func, fields in steps:
        stage = sub.add_parser(func.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    name = args.pop("action")
    {"publish": publish, **{f.__name__: f for f, _ in steps}}[name](**args)


if __name__ == "__main__":
    main()
