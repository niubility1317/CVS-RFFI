"""Match the existing R07 Residual validation panel and its frozen probes."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_practical_probe_blind as model
from . import wisig_practical_received_audit as protocol


RUN = "20261126-diagnostic-wisig-practical-probe-s2026112601-r01"
PARENT = "20261019-phase1-restoration-learned-action-source-s2026092701-r07"
COUNT = 128
SEED = 2026112601
SCENES = ("practical_high", "practical_high_urban", "practical_low_suburban",
          "practical_low_urban", "practical_mid", "practical_mid_urban")


def matched_selection():
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(1800, COUNT, replace=False)), rng


def prepare(cache, output):
    ids, _ = matched_selection()
    roles = np.load(cache / "validation/roles.npy", allow_pickle=False)
    if roles.shape != (1800,) or np.any(roles[ids] != "V"):
        raise ValueError("R07 fixed source V panel required")
    rows = []
    for scene in SCENES:
        path = cache / "validation_actions" / scene / "received.npy"
        received = np.load(path, allow_pickle=False, mmap_mode="r")
        if received.shape != (1800, 2, model.receive.SAMPLES):
            raise ValueError("Unexpected R07 receive layout")
        rows.append(received[ids, 0] + 1j * received[ids, 1])
    np.save(output, np.stack(rows), allow_pickle=False)
    print("R07_PROBE_RECEIVED_READY", len(SCENES), COUNT, flush=True)


def tune(source, cfg):
    received = np.load(source, allow_pickle=False)
    if received.shape != (12, protocol.panel.NDEV, model.receive.SAMPLES):
        raise ValueError("Expected the receive-only source U bank")
    indices = np.random.default_rng(protocol.panel.SEED).permutation(received.shape[1])
    fit, hold = np.split(indices, [96])
    prior = model.fit(received[:, fit].reshape(-1, model.receive.SAMPLES))
    np.savez(cfg, **vars(prior))
    held = received[:, hold].reshape(-1, model.receive.SAMPLES)
    times = []
    for y in held[:48]:
        start = perf_counter()
        kernels, variance = model.estimate(y[None, :], prior)
        times.append((perf_counter() - start) * 1000)
        if not np.isfinite(kernels).all() or np.any(variance <= 0.):
            raise ValueError("Nonfinite source-U smoke estimate")
    p50, p95 = np.quantile(times, [.5, .95])
    write(cfg.parent / "speed.json", dict(unit="ms/packet", threads=1, packets=len(times),
          candidate_median=float(p50), candidate_p95=float(p95), simultaneous_arms=len(model.ARMS)))
    write(cfg.parent / "configuration.json", dict(anchor=prior.anchor, physical_fit=len(fit),
          physical_hold=len(hold), bright_phase_coherence=abs(prior.phase_mean),
          coherence=prior.coherence.tolist(), noise_knots=np.exp(prior.noise_knots).tolist()))
    print("R07_PROBE_RECEIVED_PRIOR_FROZEN", prior.anchor, flush=True)


def predict(received_path, cfg, output):
    received = np.load(received_path, allow_pickle=False)
    if received.shape != (len(SCENES), COUNT, model.receive.SAMPLES):
        raise ValueError("The predictor expects only the R07 received observation matrix")
    with np.load(cfg, allow_pickle=False) as stored:
        prior = model.Prior(**{k: stored[k] for k in model.Prior.__dataclass_fields__})
    prior = model.Prior(int(prior.anchor), prior.guard, prior.decoder, prior.calibration,
                        complex(prior.phase_mean), prior.noise_knots, prior.coherence)
    result = [model.estimate(row, prior) for row in received]
    kernel, noise = zip(*result)
    np.savez(output, kernel=np.stack(kernel, axis=1), complex_variance=np.stack(noise))
    print("R07_PROBE_RECEIVED_PREDICTIONS_CLOSED", len(SCENES), COUNT, flush=True)


def filter_reflect(kernel, samples):
    index = np.r_[np.arange(model.KERNEL_LENGTH-1, 0, -1), np.arange(model.receive.SAMPLES)]
    return np.fft.ifft(np.fft.fft(samples[..., index], n=512) * np.fft.fft(kernel, n=512))[..., 63:319]


def score(root, reference):
    with np.load(root / "estimates.npz", allow_pickle=False) as stored:
        kernels, measured = stored["kernel"], stored["complex_variance"]
    shape = (len(model.ARMS), len(SCENES), COUNT, model.KERNEL_LENGTH)
    if kernels.shape != shape or measured.shape != shape[1:3] or not np.isfinite(kernels).all():
        raise ValueError("Incomplete probe estimates")
    if not np.isfinite(measured).all() or np.any(measured <= 0.):
        raise ValueError("Invalid receive-only variance estimates")
    ids, rng = matched_selection()
    zshape = (2, COUNT, model.receive.SAMPLES)
    probes = (rng.normal(size=zshape) + 1j * rng.normal(size=zshape)) / np.sqrt(2.)
    errors, variances = [], []
    for i, scene in enumerate(SCENES):
        path = reference / scene
        native = np.load(path / "kernel.npy", allow_pickle=False, mmap_mode="r")[ids].astype(float)
        curves = np.load(path / "coefficients.npy", allow_pickle=False, mmap_mode="r")[ids].astype(float)
        variance = np.load(path / "variance.npy", allow_pickle=False, mmap_mode="r")[ids]
        filtered = filter_reflect(native[:, 0] + 1j * native[:, 1], probes)
        true_output = ((curves[:, 0] + 1j * curves[:, 1]) * filtered +
                       (curves[:, 2] + 1j * curves[:, 3]) * filtered.conj())
        predicted = filter_reflect(kernels[:, i, None], probes[None, :])
        power = np.mean(abs(true_output)**2, axis=-1).clip(1e-4)
        errors.append((np.mean(abs(predicted - true_output)**2, axis=-1) / power).mean(axis=1))
        variances.append(abs(10. * np.log10(measured[i] / (2. * variance))))
    e, v = np.asarray(errors), np.asarray(variances)
    if not np.isfinite(e).all() or not np.isfinite(v).all():
        raise ValueError("Invalid matched R07 probe or variance result")
    def summarize(x, n):
        return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.quantile(x, .90)),
                    p95=float(np.quantile(x, .95)), noise_median=float(np.median(n)),
                    noise_p95=float(np.quantile(n, .95)))
    results = {arm: dict(aggregate=summarize(e[:, i], v),
               scenes={scene: summarize(e[g, i], v[g]) for g, scene in enumerate(SCENES)})
               for i, arm in enumerate(model.ARMS)}
    write(root / "scores.json", dict(status="ANALYZED", receive_only=True, truth_side_last=True,
          reference="R07 validation_actions", probes=2, packets=COUNT, probe_seed=SEED,
          settings=read(root / "configuration.json"), speed=read(root / "speed.json"), results=results))
    print("R07_PROBE_RECEIVED_SCORED", len(SCENES), COUNT, flush=True)


def smoke():
    rng = np.random.default_rng(29)
    probe = rng.normal(size=model.receive.SAMPLES) + 1j * rng.normal(size=model.receive.SAMPLES)
    k = np.zeros(model.KERNEL_LENGTH, complex)
    k[17] = .5 + .1j
    np.testing.assert_allclose(filter_reflect(k, probe), k[17] * probe[np.abs(model.TIME-17)], atol=1e-12)
    print("R07_PROBE_RECEIVED_SMOKE_PASS", flush=True)


def dispatch(runtime):
    rt = read(runtime)
    root, cache, previous = (Path(rt[k]) for k in ("run_root", "cache", "previous"))
    cfg, obs = root / "configuration.npz", root / "evaluation_received.npy"
    steps = (("smoke", ()), ("prepare", (cache, obs)), ("tune", (previous / "source_received.npy", cfg)),
             ("predict", (obs, cfg, root / "estimates.npz")), ("score", (root, cache / "validation_actions")))
    protocol.panel.full.source.parent.parent.prior_runner.dispatch(rt, steps)


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    runs = PurePosixPath(ctx["project"]) / "runs"
    inputs = dict(cache=str(runs / PARENT / "cache"), previous=str(runs / protocol.RUN))
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
    steps = ((smoke, ()), (prepare, ("cache", "output")), (tune, ("source", "cfg")),
             (predict, ("received_path", "cfg", "output")), (score, ("root", "reference")),
             (dispatch, ("runtime",)))
    for f, fields in steps:
        stage = sub.add_parser(f.__name__)
        for name in fields:
            stage.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    name = args.pop("action")
    {"publish": publish, **{f.__name__: f for f, _ in steps}}[name](**args)


if __name__ == "__main__":
    main()
