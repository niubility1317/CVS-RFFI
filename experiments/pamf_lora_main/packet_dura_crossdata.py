"""Paired source test of a Packet U-Net prior and DURA observation consistency."""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


RUN = "20261127-phase1-lora-wisig-packet-dura-s2026113021-r01"
RETRY_RUN = RUN.removesuffix("r01") + "r02"
LORA_PARENT = "20261127-phase1-lora-raw8192-e104-strong-s2026113011-r01"
LORA_DURA = "20261127-phase1-lora-raw8192-r02-r07-r17-s2026113009-r01"
WISIG_REFERENCE = "20261125-phase1-wisig-native-fir-source-m4-r01"
SEED, STEPS, BATCH = 2026113021, 1600, 24
MODES = ("direct", "dura", "packet_unet", "hybrid")
VIEWS = ("clean", "practical_high", "practical_mid", "practical_low_suburban",
         "practical_high_urban", "practical_mid_urban", "practical_low_urban")
DEV = "cuda:0"


def rms(x):
    return x.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


def blend(y, proposal, action, variance):
    """Reconcile a learned prior with the observation, limiting extra IQ edits."""
    scale = rms(y)
    y_normalized, prior = y / scale, proposal / scale
    solved = weighted_map(action, y_normalized, prior, variance.reshape(len(y)) + .02, steps=2)
    change = solved - prior
    budget = .5 * (prior - y_normalized).abs().square().mean(-1, keepdim=True).sqrt()
    ratio = (budget / rms(change)).clamp_max(1.)
    return proposal + scale * ratio * change


def archive_imports(config, *, extra=()):
    """Resolve each historical reader against the release that produced it."""
    from experiments.pamf_lora_main.runner import activate
    roots = [str(Path(x) / "experiments") for x in (*extra, config["release_root"])]
    activate(config)
    package = importlib.import_module("experiments")
    package.__path__[:0] = roots
    # The launcher's subpackage is already imported by python -m.
    lora = importlib.import_module("experiments.pamf_lora_main")
    lora.__path__[:0] = [str(Path(root) / "pamf_lora_main") for root in roots]


def reference(rt, dataset):
    key = "lora_ref" if dataset == "lora" else "wisig_ref"
    config = read(Path(rt[key]) / "runtime.json")
    if dataset == "lora":
        expected = LORA_PARENT, "39be84764f9c1340823f4e9d94df958911307056"
    else:
        expected = WISIG_REFERENCE, "ef9f4c227ce57c79dc17a3aeb599395cc25d883b"
    if (config["run_id"], config["code_commit"]) != expected:
        raise ValueError("Unexpected frozen source reference")
    return config


def lora_deps(config):
    archive_imports(config, extra=(config["w24_release_root"],))
    from experiments.pamf_lora_main import w24_joint_pamf as joint
    from experiments.pamf_lora_main import w24_native_training as native
    return joint, native


def wisig_deps(config):
    archive_imports(config)
    from experiments.pamf_unrolled_restore import wisig_anchored_window_source as original
    return original


def wifi_proposal(model, raw):
    groups = raw.unflatten(-1, (8, 32))
    scale = rms(groups)
    y = groups / scale
    quality = (y[..., 1:] * y[..., :-1].conj()).mean(-1, keepdim=True).abs().clamp_max(1.)
    edges = torch.cat((groups[..., :24], groups[..., -24:]), dim=-1)
    corrected = model(y, quality, scale, edges)[-1]
    return raw + .5 * ((corrected - y) * scale).flatten(-2)


def disturb(clean, rng):
    # Generic echoes and white noise, with no practical-view parameters.
    def draw():
        return torch.rand((len(clean), 1), generator=rng, device=clean.device)

    echo = torch.randn((len(clean), 1), generator=rng, device=clean.device, dtype=clean.dtype)
    delayed = torch.nn.functional.pad(clean[:, :-3], (3, 0))
    noise = torch.randn(clean.shape, generator=rng, device=clean.device, dtype=clean.dtype)
    level = (10. ** (-(-8. + 32. * draw()) / 20.))
    return clean + .15 * echo * delayed + level * noise / rms(noise)


def fit_wifi(original, source_root, folder):
    data, ids, _ = original.core.source_iq(Path(source_root))
    torch.manual_seed(SEED)
    model = PacketUNet(across_chirps=True, center=32).to(DEV).train()
    rng = np.random.default_rng(SEED)
    perturb = torch.Generator(device=DEV).manual_seed(SEED + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    for step in range(1, STEPS + 1):
        x = complex_iq(tensor(data[rng.choice(ids, BATCH, replace=False)], DEV))
        x = x / rms(x)
        y = disturb(x, perturb)
        error = wifi_proposal(model, y) - x
        clean_edit = wifi_proposal(model, x) - x
        power = torch.fft.fft(x, norm="ortho").abs().square()
        spectral = (torch.fft.fft(error, norm="ortho").abs().square()
                    / (.03 + power)).log1p().mean()
        loss = error.abs().square().mean() + .3 * spectral + 1.5 * clean_edit.abs().square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 1 and (norm <= 0 or not torch.isfinite(loss)):
            raise ValueError("WiSig Packet U-Net training path is inactive")
        optimizer.step()
        schedule.step()
        if step == 1 or step % 100 == 0:
            print("WISIG_PACKET_FIT", step, float(loss.detach()), flush=True)
    with (folder / "packet_unet.pt").open("xb") as stream:
        torch.save(model.state_dict(), stream)
    write(folder / "fit.json", dict(status="SOURCE_FROZEN", steps=STEPS, seed=SEED,
                                   U_count=len(ids), TX_reads=0, target_reads=0, F_reads=0))
    return model.eval().requires_grad_(False)


def freeze(folder, ids, matrix, archive=None):
    ids = np.asarray(ids)
    if (set(matrix) != set(VIEWS) or len(ids) != len(set(ids.tolist()))
            or any(m.shape != (len(MODES), len(ids)) for m in matrix.values())):
        raise ValueError("Incomplete physical ID and seven-view predictions")
    with (folder / "predictions.npz").open("xb") as stream:
        np.savez(stream, ids=ids, **matrix)
    write(folder / "frozen.json", dict(status="PREDICTIONS_FROZEN", modes=list(MODES),
                                     views=list(VIEWS), n=len(ids), source_truth_reads=0,
                                     target_reads=0, archived_comparator=archive))


@torch.no_grad()
def predict_lora(config, folder):
    joint, native = lora_deps(config)
    source = Path(config["source_root"])
    contract, reader, _, _ = joint.boot(config, DEV)
    task_root = Path(config["run_root"])
    dura_root = task_root.parent / LORA_DURA
    proposal = native.task_model(None, DEV).eval().requires_grad_(False)
    native.load_stage(task_root, native.TASK, contract, proposal.core, DEV)
    dura = native.dura_model(dura_root, contract, DEV, stage="raw_r07").eval().requires_grad_(False)

    def classify(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        return joint.classify_aperture(reader, joint.received_aperture(cpu, DEV))

    def apply(x):
        y = tensor(np.ascontiguousarray(x).view(np.float32), DEV).view(torch.complex64)
        p = proposal(y)
        _, action, variance = dura.estimate(pair(y / rms(y)))
        restored = blend(y, p, action, variance)
        return np.stack([classify(z) for z in (y, p, restored)])

    x = joint.source_role(source, contract, "U_s")[0]
    smoke = apply(np.stack((x[0], joint.source_receive(VIEWS[-1], (0, joint.SOURCE[0], x[1])))))
    if smoke.shape != (3, 2):
        raise ValueError("LoRa frozen inference smoke failed")
    write(folder / "smoke.json", dict(status="PASS", n=2, source_V_reads=0, target_reads=0))
    print("LORA_REAL_SMOKE_PASS", flush=True)
    task_panel, dura_panel = [r / "full_native" for r in (task_root, dura_root)]
    records = [read(p / "source-predictions.json") for p in (task_panel, dura_panel)]
    indices, receivers, ids = joint.panel(source, contract)
    if any((r["status"], r["physical_ids"], r["views"], r["source_truth_reads"], r["target_reads"])
           != ("PREDICTIONS_FROZEN", ids, list(VIEWS), 0, 0) for r in records):
        raise ValueError("LoRa archived same-panel controls differ")
    raw = joint.native(source, "V", contract["counts"]["V"])[indices]
    jobs = [(int(i), joint.SOURCE[int(r)], x) for i, r, x in zip(indices, receivers, raw, strict=True)]
    matrix = {}
    with np.load(task_panel / "predictions.npz", allow_pickle=False) as task, \
         np.load(dura_panel / "predictions.npz", allow_pickle=False) as control:
        for view in VIEWS:
            views = np.stack([joint.source_receive(view, item) for item in jobs])
            fresh = np.concatenate([apply(views[start:start + joint.PRED_BATCH])
                                    for start in range(0, len(views), joint.PRED_BATCH)], axis=1)
            fixed = task[view][[records[0]["modes"].index(m) for m in ("direct", native.TASK)]]
            if not np.array_equal(fresh[:2], fixed):
                raise ValueError(f"LoRa direct or Packet U-Net prediction differs from archive: {view}")
            d = control[view][records[1]["modes"].index("raw_r17")]
            matrix[view] = np.stack((fresh[0], d, fresh[1], fresh[2]))
            print("LORA_PACKET_DURA", view, len(ids), flush=True)
    freeze(folder, ids, matrix, archive=LORA_DURA)


def predict_wifi(config, folder):
    original = wisig_deps(config)
    data, _, calibration = original.core.source_iq(Path(config["source_root"]))
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    parent = Path(config["parent_root"])
    engine = original.tri.core.original_r69(parent)
    # Use the same validated R69 model for the comparator and action estimate.
    base = engine.pipeline.model

    @torch.no_grad()
    def apply(y, proposal):
        p = wifi_proposal(proposal, y)
        _, action, variance = base.estimate_context(pair(y / rms(y)))
        restored = blend(y, p, action, variance)
        (d,) = engine(pair(y))
        return np.asarray([original.classify(reader, z).argmax(1).tolist()
                           for z in (y, complex_iq(d), p, restored)], dtype=np.int16)

    sample = complex_iq(tensor(data[calibration[:4]], DEV))
    probe = PacketUNet(across_chirps=True, center=32).to(DEV).eval()
    if apply(sample, probe).shape != (4, 4):
        raise ValueError("WiSig frozen inference smoke failed")
    write(folder / "smoke.json", dict(status="PASS", n=4, source_V_reads=0, target_reads=0))
    print("WISIG_REAL_SMOKE_PASS", flush=True)
    # The fit routine sees only the TX-hidden training cache and output path.
    proposal = fit_wifi(original, config["source_root"], folder)
    cache, _, _, _, ids, _, _ = original.tri.core.handles(config)
    matrix = {}
    panel = cache / "validation_full"
    for view in VIEWS:
        iq = np.load(panel / f"{view}.npy", mmap_mode="r", allow_pickle=False)
        if iq.shape != (len(ids), 2, 256) or iq.dtype != np.float32:
            raise ValueError("WiSig validated view no longer matches the physical IDs")
        matrix[view] = np.concatenate([apply(complex_iq(tensor(iq[start:start+48], DEV)), proposal)
                                       for start in range(0, len(ids), 48)], axis=1)
        print("WISIG_PACKET_DURA", view, len(ids), flush=True)
    freeze(folder, ids, matrix)


def score(rt):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Scoring requires an independent CPU process")
    root = Path(rt["run_root"])
    matrices = {}
    for dataset in ("lora", "wisig"):
        folder = root / dataset
        marker = read(folder / "frozen.json")
        if any(marker[k] != v for k, v in dict(status="PREDICTIONS_FROZEN", modes=list(MODES),
                views=list(VIEWS), source_truth_reads=0, target_reads=0).items()):
            raise ValueError("Both complete truth-blind prediction matrices are required")
        with np.load(folder / "predictions.npz", allow_pickle=False) as rows:
            ids = rows["ids"]
            pred = np.stack([rows[name] for name in VIEWS])
        if len(ids) != marker["n"] or pred.shape != (7, 4, len(ids)):
            raise ValueError("Frozen prediction shape differs")
        matrices[dataset] = ids, pred

    lora = reference(rt, "lora")
    source = Path(lora["source_root"])
    ids, pred = matrices["lora"]
    with np.load(source / "V_metadata.npz", allow_pickle=False) as meta:
        positions = {value: index for index, value in enumerate(meta["ids"].tolist())}
        order = [positions[key] for key in ids]
        lora_score = summarize(pred, meta["y"][order], meta["rx_i"][order])

    wifi = reference(rt, "wisig")
    original = wisig_deps(wifi)
    _, _, source, _, panel_ids, rx, ordering = original.tri.core.handles(wifi)
    ids, pred = matrices["wisig"]
    if not np.array_equal(panel_ids, ids):
        raise ValueError("WiSig physical IDs differ from the frozen evaluation panel")
    truth = np.load(source / "V/labels.npy", allow_pickle=False)[ordering]
    wifi_score = summarize(pred, truth, rx)
    write(root / "source-scores.json", dict(status="SOURCE_SCORED", truth_last=True, target_reads=0,
          modes=list(MODES), views=list(VIEWS), lora=lora_score, wisig=wifi_score,
          both_qualify=lora_score["qualifies"] and wifi_score["qualifies"]))


def summarize(pred, truth, receivers):
    hit = pred == truth
    correct = hit.sum(axis=-1)
    practical = correct[1:].sum(axis=0)
    n = len(truth)
    rng = np.random.default_rng(SEED + 2)
    samples = [np.flatnonzero(receivers == r) for r in np.unique(receivers)]
    boot = np.concatenate([rng.choice(group, (2000, len(group))) for group in samples], axis=1)
    contrasts = {}
    for i, name in enumerate(MODES[:-1]):
        change = hit[:, -1].astype(np.int8) - hit[:, i].astype(np.int8)
        contrasts[name] = {}
        for key, data in (("clean", change[:1]), ("six", change[1:]), ("low_urban", change[-1:])):
            contrasts[name][key] = dict(repaired=int((data > 0).sum()), harmed=int((data < 0).sum()),
                net=int(data.sum()), ci95=np.quantile(data.sum(axis=0)[boot].sum(axis=1), [.025, .975]).tolist(),
                per_receiver={str(r): int(data[:, receivers == r].sum()) for r in np.unique(receivers)})
    return dict(n=n, by_scene={mode: dict(zip(VIEWS, correct[:, i].tolist(), strict=True))
                               for i, mode in enumerate(MODES)},
                six=dict(zip(MODES, practical.tolist(), strict=True)), contrasts=contrasts,
                qualifies=bool(practical[-1] > practical[:-1].max()
                               and correct[-1, -1] >= correct[-1, :-1].max()
                               and correct[0, -1] >= correct[0, :-1].max()))


def dispatch(runtime, rt):
    from experiments.pamf_lora_main.runner import available
    folder = Path(rt["run_root"])
    folder.mkdir()
    state = folder / "state.json"

    def launch(dataset, card):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if card is None else str(card),
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        with (folder / f"{dataset}.log").open("x", encoding="utf-8") as stream:
            worker = subprocess.Popen([sys.executable, "-u", "-m", __spec__.name, dataset, str(runtime)],
                                      stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, env=env)
        print("PACKET_DURA_WORKER", dataset, worker.pid, card, flush=True)
        return worker

    try:
        deadline = monotonic() + 900
        while not (cards := available(rt["allowed_gpus"], max_compute_processes=8)):
            if monotonic() >= deadline:
                raise RuntimeError("No N607 GPU with 12 GiB free")
            sleep(10)
        devices = dict(zip(("lora", "wisig"), (cards * 2)[:2], strict=True))
        workers = {name: launch(name, card) for name, card in devices.items()}
        write(state, dict(status="RUNNING", workers={k: dict(pid=p.pid, gpu=devices[k])
                                                   for k, p in workers.items()}))
        exits = {name: worker.wait() for name, worker in workers.items()}
        if any(exits.values()):
            raise RuntimeError(f"Source worker exits: {exits}")
        if launch("score", None).wait():
            raise RuntimeError("Independent source scorer failed")
        write(state, dict(status="SOURCE_SCORED", both_qualify=read(folder/"source-scores.json")["both_qualify"]),
              exclusive=False)
    except Exception as error:
        write(state, dict(status="FAILED", error=str(error)), exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("publish", "dispatch", "lora", "wisig", "score"))
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ssh-config", type=Path)
    args = parser.parse_args()
    rt = read(args.runtime)
    if args.action == "publish":
        from experiments.pamf_lora_main.publish import release, release_commit
        project = rt["project"]
        release_root = f"{project}/releases"
        rt.update(run_id=RETRY_RUN, code_commit=release_commit(Path.cwd()), runner=__spec__.name,
                  release_root=f"{release_root}/{RETRY_RUN}-release", run_root=f"{project}/runs/{RETRY_RUN}",
                  lora_ref=f"{release_root}/{LORA_PARENT}-release",
                  wisig_ref=f"{release_root}/{WISIG_REFERENCE}-release",
                  allowed_gpus=[7, 2, 0, 6, 3, 5])
        rt["external_releases"] = [rt["lora_ref"], rt["wisig_ref"], f"{project}/runs/{LORA_DURA}"]
        release(Path.cwd(), args.output, rt, args.ssh_config)
    elif args.action == "dispatch":
        dispatch(args.runtime, rt)
    elif args.action == "score":
        score(rt)
    else:
        numerical_policy()
        folder = Path(rt["run_root"]) / args.action
        folder.mkdir()
        (predict_lora if args.action == "lora" else predict_wifi)(reference(rt, args.action), folder)


if __name__ == "__main__":
    main()
