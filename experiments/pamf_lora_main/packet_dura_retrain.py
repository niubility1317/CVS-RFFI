"""Retrain the learned proposal and the DURA observation estimate together."""

import argparse
from pathlib import Path
import os
import subprocess
import sys
from time import monotonic, sleep

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main import packet_dura_crossdata as replay


RUN = "20261127-phase1-lora-wisig-packet-dura-retrain-s2026113025-r01"
SEED = 2026113025
STEPS = 1500
WIDTHS = {"lora": 1024, "wisig": 32}
BATCHES = {"lora": 6, "wisig": 24}
DEV = "cuda:0"


class JointRecovery(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.proposal = replay.PacketUNet(across_chirps=True, center=width)
        self.context = nn.Sequential(nn.Conv1d(4, 24, 9, stride=4, padding=4), nn.SiLU(),
                                     nn.Conv1d(24, 32, 7, stride=2, padding=3), nn.SiLU())
        self.action = nn.Linear(32, 37)
        nn.init.zeros_(self.action.weight)
        nn.init.zeros_(self.action.bias)

    def forward(self, received):
        scale = replay.rms(received)
        y = received / scale
        groups = y.unflatten(-1, (8, self.width))
        group_scale = replay.rms(groups)
        normal = groups / group_scale
        coherence = (normal[..., 1:] * normal[..., :-1].conj()).mean(-1, keepdim=True).abs().clamp_max(1.)
        edges = torch.cat((groups[..., :24], groups[..., -24:]), dim=-1)
        predicted = self.proposal(normal, coherence, group_scale, edges)[-1]
        p = y + .5 * ((predicted - normal) * group_scale).flatten(-2)

        # One RX-side estimate for each physical record. No simulated action is supplied.
        features = self.context(torch.cat((replay.pair(y), replay.pair(p)), dim=1)).mean(-1)
        taps_r, taps_i, phase, slope, image_r, image_i, log_noise = self.action(features).split((16, 16, 1, 1, 1, 1, 1), dim=1)
        taps = .12 * torch.complex(taps_r.tanh(), taps_i.tanh())
        impulse = F.pad(torch.ones_like(taps[:, :1]), (0, 15))
        h = taps + impulse
        t = torch.linspace(-1, 1, y.shape[-1], device=y.device)
        rotation = torch.polar(torch.ones_like(y.real), .5 * phase.tanh() + slope.tanh() * t)
        image = .12 * torch.complex(image_r.tanh(), image_i.tanh()).expand_as(y)
        v = .004 + .696 * (log_noise - 1.8).sigmoid()
        action = Action(h, rotation, image)
        posterior = replay.blend(y, p, action, v.detach())
        return p * scale, posterior * scale, action, v, scale


def perturb(clean, rng):
    opts = dict(device=clean.device, generator=rng)
    b, length = clean.shape

    def draw():
        return torch.rand((b, 1), **opts)

    echo = torch.randn((b, 1), dtype=clean.dtype, **opts)
    image = torch.randn((b, 1), dtype=clean.dtype, **opts)
    time = torch.linspace(-1, 1, length, device=clean.device)
    phase = .3 * (2 * draw() - 1) + .6 * (2 * draw() - 1) * time
    shifted = clean + .10 * echo * F.pad(clean[:, :-3], (3, 0))
    distorted = torch.polar(torch.ones_like(clean.real), phase) * shifted + .025 * image * shifted.conj()
    noise = torch.randn(clean.shape, dtype=clean.dtype, **opts)
    strong = draw() < 1 / 3
    snr = torch.where(strong, -8 + 12 * draw(), 4 + 20 * draw())
    return distorted + 10. ** (-snr / 20) * noise / replay.rms(noise)


def wave_error(value, clean):
    delta = value - clean
    power = torch.fft.fft(clean, norm="ortho").abs().square()
    spectral = (torch.fft.fft(delta, norm="ortho").abs().square()
                / (.03 + power)).log1p().mean()
    return delta.abs().square().mean() + .30 * spectral


def step_loss(model, reference, rng):
    clean = reference / replay.rms(reference)
    y = perturb(clean, rng)
    p, restored, action, variance, scale = model(y)
    clean_p, clean_out, *_ = model(clean[:max(1, len(clean) // 4)])
    clean_reference = clean[:len(clean_p)]
    # Pairwise observation fitting uses each generated y and the source waveform.
    # Detach each complementary term so the learned variance cannot lower the
    # restoration loss simply by turning off the observation update.
    residual = (action(clean / scale) - y / scale).abs().square()
    physical_error = residual.mean()
    variance_loss = (.5 * residual.detach().mean(-1, keepdim=True) / variance + variance.log()).mean()
    identity = .5 * ((clean_p - clean_reference).abs().square().mean()
                     + (clean_out - clean_reference).abs().square().mean())
    terms = (wave_error(restored, clean), wave_error(p, clean), physical_error, variance_loss, identity)
    return terms[0] + .5 * terms[1] + .1 * terms[2] + .01 * terms[3] + 1.5 * terms[4], terms


def training_source(dataset, rt):
    config = replay.reference(rt, dataset)
    source = Path(config["source_root"])
    if dataset == "lora":
        joint, _ = replay.lora_deps(config)
        data = joint.source_role(source, joint.source_contract(source), "U_s")[0]
        return data, np.arange(len(data))
    original = replay.wisig_deps(config)
    data, ids, _ = original.core.source_iq(source)
    return data, ids


def complex_batch(array):
    if np.iscomplexobj(array):
        return tensor(np.ascontiguousarray(array).view(np.float32), DEV).view(torch.complex64)
    return replay.complex_iq(tensor(array, DEV))


def fit(rt, dataset, folder):
    data, ids = training_source(dataset, rt)
    torch.manual_seed(SEED)
    model = JointRecovery(WIDTHS[dataset]).to(DEV).train()
    rng = np.random.default_rng(SEED + 1)
    perturbations = torch.Generator(device=DEV).manual_seed(SEED + 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    started = monotonic()
    for step in range(1, STEPS + 1):
        x = complex_batch(data[rng.choice(ids, BATCHES[dataset], replace=False)])
        loss, terms = step_loss(model, x, perturbations)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 1:
            grads = [model.proposal.spectral[-1].weight.grad, model.action.weight.grad]
            if not torch.isfinite(loss) or any(g is None or not bool(g.norm() > 0) for g in grads):
                raise ValueError("Real source smoke showed inactive or nonfinite joint gradients")
            write(folder / "fit-smoke.json", dict(status="PASS", gradient=float(gradient),
                                                proposal_and_action_gradients=True, target_reads=0))
        optimizer.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print("JOINT_RETRAIN", dataset, step, *[float(x.detach()) for x in terms], flush=True)
    with (folder / "joint.pt").open("xb") as stream:
        torch.save(model.state_dict(), stream)
    write(folder / "fit.json", dict(status="SOURCE_FROZEN", dataset=dataset, steps=STEPS,
                                   seed=SEED, batch=BATCHES[dataset], source_u_pool=len(ids),
                                   elapsed_seconds=monotonic()-started,
                                   params=sum(p.numel() for p in model.parameters()),
                                   peak_cuda=torch.cuda.max_memory_allocated(),
                                   F_reads=0, source_V_reads=0, TX_reads=0, target_reads=0))


@torch.no_grad()
def predict(rt, dataset, folder):
    model = JointRecovery(WIDTHS[dataset]).to(DEV).eval().requires_grad_(False)
    if read(folder / "fit.json")["status"] != "SOURCE_FROZEN":
        raise ValueError("Joint training checkpoint must be fixed before prediction")
    model.load_state_dict(torch.load(folder / "joint.pt", map_location=DEV, weights_only=True), strict=True)
    config = replay.reference(rt, dataset)
    source = Path(config["source_root"])
    archive_root = Path(rt["project"]) / "runs"
    archive = archive_root / (replay.RUN if dataset == "lora" else replay.WISIG_RUN) / dataset
    marker = read(archive / "frozen.json")
    if any(marker[k] != v for k, v in dict(status="PREDICTIONS_FROZEN", modes=list(replay.MODES),
            views=list(replay.VIEWS), target_reads=0, source_truth_reads=0).items()):
        raise ValueError("Expected matching frozen replay comparator rows")
    matrix = {}
    if dataset == "lora":
        joint, _ = replay.lora_deps(config)
        contract, reader, _, _ = joint.boot(config, DEV)
        training_iq = joint.source_role(source, contract, "U_s")[0][:2]

        def panel():
            rows, receivers, ids = joint.panel(source, contract)
            raw = joint.native(source, "V", contract["counts"]["V"])[rows]
            items = [(int(i), joint.SOURCE[int(rx)], x)
                     for i, rx, x in zip(rows, receivers, raw, strict=True)]
            return ids, items

        def get_iq(view):
            return np.stack([joint.source_receive(view, item) for item in items])

        def classify(signal):
            cpu = np.asarray(signal.contiguous().view(torch.float32).tolist(), dtype=np.float32)
            return joint.classify_aperture(reader, joint.received_aperture(cpu.view(np.complex64), DEV))

        batch = joint.PRED_BATCH
    else:
        original = replay.wisig_deps(config)
        reader = original.tri.reader_factory(config).eval().requires_grad_(False)
        data, _, calibration = original.core.source_iq(source)
        training_iq = data[calibration[:2]]

        def panel():
            cache, _, _, _, ids, _, _ = original.tri.core.handles(config)
            return ids, cache

        def get_iq(view):
            return np.load(cache / "validation_full" / f"{view}.npy", mmap_mode="r", allow_pickle=False)

        def classify(signal):
            return np.asarray(original.classify(reader, signal).argmax(1).tolist(), dtype=np.int16)

        batch = 48

    def readout(x):
        p, recovered, *_ = model(complex_batch(x))
        return np.stack((classify(p), classify(recovered)))

    if readout(training_iq).shape != (2, 2):
        raise ValueError("Real source checkpoint with frozen classifier failed")
    write(folder / "smoke.json", dict(status="PASS", input_records=2, source_V_reads=0, target_reads=0))
    print("RETRAIN_READER_SMOKE", dataset, flush=True)
    ids, panel_payload = panel()
    if dataset == "lora":
        items = panel_payload
    else:
        cache = panel_payload
    with np.load(archive / "predictions.npz", allow_pickle=False) as old:
        if not np.array_equal(old["ids"], np.asarray(ids)):
            raise ValueError("Truth-blind controls and retrained readout physical IDs differ")
        for view in replay.VIEWS:
            iq = get_iq(view)
            learned = np.concatenate([readout(iq[i:i+batch]) for i in range(0, len(ids), batch)], axis=1)
            matrix[view] = np.concatenate((old[view][:2], learned), axis=0)
            print("RETRAIN_FROZEN_VIEW", dataset, view, len(ids), flush=True)
    replay.freeze(folder, ids, matrix, archive=str(archive))


def dispatch(runtime, rt):
    from experiments.pamf_lora_main.runner import available
    folder = Path(rt["run_root"])
    folder.mkdir()
    state = folder / "state.json"

    def start(role, gpu, dataset=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        name = role if dataset is None else f"{dataset}-{role}"
        command = [sys.executable, "-u", "-m", __spec__.name, role, str(runtime)]
        if dataset is not None:
            command.append(dataset)
        with (folder / f"{name}.log").open("x", encoding="utf-8") as stream:
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                      stderr=subprocess.STDOUT, env=env)
        print("JOINT_WORKER", name, worker.pid, gpu, flush=True)
        return worker

    try:
        gpu_order = rt["allowed_gpus"]
        deadline = monotonic() + 900
        while len(cards := available(gpu_order, max_compute_processes=8)) < 2:
            if monotonic() >= deadline:
                raise RuntimeError("Two available GPUs required for parallel source training")
            sleep(12)
        devices = dict(zip(WIDTHS, cards[:2], strict=True))
        workers = {}
        for dataset, card in devices.items():
            (folder / dataset).mkdir()
            workers[dataset] = start("fit", card, dataset)
        write(state, dict(status="TRAINING", gpu=devices, fit_pid={k: v.pid for k, v in workers.items()}))
        # Each dataset advances to frozen prediction when its own fit completes.
        predicted = {}
        while workers or predicted:
            for name, worker in list(workers.items()):
                status = worker.poll()
                if status is None:
                    continue
                del workers[name]
                if status:
                    raise RuntimeError(f"{name} fit exited {status}")
                predicted[name] = start("predict", devices[name], name)
            for name, worker in list(predicted.items()):
                status = worker.poll()
                if status is not None:
                    del predicted[name]
                    if status:
                        raise RuntimeError(f"{name} prediction exited {status}")
            write(state, dict(status="RUNNING", gpu=devices,
                fit_pid={k: v.pid for k, v in workers.items()},
                predict_pid={k: v.pid for k, v in predicted.items()}), exclusive=False)
            if workers or predicted:
                sleep(3)
        if start("score", None).wait():
            raise RuntimeError("Independent scorer failed")
        write(state, dict(status="SOURCE_SCORED", both_qualify=read(folder/"source-scores.json")["both_qualify"]),
              exclusive=False)
    except Exception as error:
        write(state, dict(status="FAILED", error=str(error)), exclusive=False)
        raise


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("role", choices=("publish", "dispatch", "fit", "predict", "score"))
    cli.add_argument("runtime", type=Path)
    cli.add_argument("dataset", choices=tuple(WIDTHS), nargs="?")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--ssh-config", type=Path)
    args = cli.parse_args()
    rt = read(args.runtime)
    if args.role == "publish":
        from experiments.pamf_lora_main.publish import release, release_commit
        project = rt["project"]
        releases = f"{project}/releases"
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(Path.cwd()),
                  release_root=f"{releases}/{RUN}-release", run_root=f"{project}/runs/{RUN}",
                  lora_ref=f"{releases}/{replay.LORA_PARENT}-release",
                  wisig_ref=f"{releases}/{replay.WISIG_REFERENCE}-release",
                  allowed_gpus=[7, 2, 6, 0, 3, 5, 1, 4])
        rt["external_releases"] = [rt["lora_ref"], rt["wisig_ref"], f"{project}/runs/{replay.RUN}",
                                   f"{project}/runs/{replay.WISIG_RUN}"]
        release(Path.cwd(), args.output, rt, args.ssh_config)
    elif args.role == "dispatch":
        dispatch(args.runtime, rt)
    elif args.role == "score":
        replay.score(rt)
    else:
        numerical_policy()
        folder = Path(rt["run_root"]) / args.dataset
        (fit if args.role == "fit" else predict)(rt, args.dataset, folder)


if __name__ == "__main__":
    main()
