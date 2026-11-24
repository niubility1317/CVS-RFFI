"""Frozen PAMF+DURA front-end and channel robustness experiment."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.cric_dura_comparison.evaluation import VIEWS, handle, metrics, reject_truth
from experiments.cric_dura_comparison.io import read, tensor, write
from .historical import write_csv

RUN = "20261124-phase1-pamf-dura-advisor-s2026092701-r01"
SEED = 20261124
COUNT = 3072
TASKS = ("cross_rx", "cross_both")
METHODS = ("raw", "pilot_fir_mmse", "coarse", "one_stage", "dura")
CHANNELS = (
    "noise14_mid_urban", "cfo15k_mid_urban", "delay4x_mid_urban", "rayleigh_tdl",
)
PANELS = (*VIEWS, *(f"fresh_{v}" for v in VIEWS[1:]), *CHANNELS)


class PilotEqualizer:
    """Packet-local L-STF residual CFO, regularized FIR estimate and MMSE inverse."""

    def __init__(self, device):
        from experiments.cvs_rff_physics.known_excitation import TONES, template
        self.reference = torch.tensor(template(np.arange(20)).tolist(), device=device, dtype=torch.complex64)
        self.bin20 = torch.tensor((TONES // 4 % 20).tolist(), device=device)
        self.sample = torch.arange(256, dtype=torch.float32, device=device)
        tones = torch.tensor(TONES.tolist(), dtype=torch.float32, device=device) / 80
        taps = torch.arange(9, dtype=torch.float32, device=device) - 4
        basis = torch.exp(-2j * torch.pi * tones[:, None] * taps)
        gram = basis.conj().T @ basis
        self.fit = torch.linalg.solve(gram + .05 * torch.eye(9, device=device), basis.conj().T)
        fft_freq = torch.fft.fftfreq(256, device=device)
        self.full_basis = torch.exp(-2j * torch.pi * fft_freq[:, None] * taps)
        self.pilot = torch.fft.fft(self.reference)[self.bin20]

    def __call__(self, iq):
        z = torch.complex(iq[:, 0], iq[:, 1])
        w = z[:, 80:160]
        lag = (w[:, 20:] * w[:, :-20].conj()).sum(1)
        omega = torch.angle(lag) / 20
        z = z * torch.exp(-1j * omega[:, None] * (self.sample - 80))
        repeats = z[:, 80:160].reshape(-1, 4, 20)
        pilot = repeats.mean(1)
        h = torch.fft.fft(pilot)[:, self.bin20] / self.pilot
        h = (h @ self.fit.T) @ self.full_basis.T
        noise = (repeats - pilot[:, None]).abs().square().mean((1, 2))
        normalized_noise = noise / pilot.abs().square().mean(1).clamp_min(1e-6)
        inverse = h.conj() / (h.abs().square() + normalized_noise[:, None] + 1e-3)
        inverse = inverse / (inverse.abs() / 4).clamp_min(1)
        result = torch.fft.ifft(torch.fft.fft(z) * inverse)
        return torch.stack((result.real, result.imag), dim=1)


def simulated(clean, ids, panel):
    from leo_practical.channel import Config, receiver_for_session, stable_seed, normalize_input
    from leo_practical.residual import ResidualChannel
    result = np.empty_like(clean)
    scene = panel.removeprefix("fresh_") if panel.startswith("fresh_") else "practical_mid_urban"
    nominal = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route="residual")
    cfg = nominal
    if panel == "noise14_mid_urban":
        cfg = replace(nominal, reference_snr_db=14.)
    elif panel == "cfo15k_mid_urban":
        cfg = replace(nominal, residual_cfo_std_hz=15000.)
    elif panel == "delay4x_mid_urban":
        cfg = replace(nominal, rms_delay_ns=tuple(4 * x for x in nominal.rms_delay_ns))
    # A fixed simulated receiver profile is shared by all fresh/stress cells.
    rx = receiver_for_session(cfg, SEED, "advisor-frozen-rx")
    for i, packet_id in enumerate(ids):
        x, _ = normalize_input(clean[i, 0].astype(np.float64) + 1j * clean[i, 1].astype(np.float64))
        seed = stable_seed(SEED, "advisorPracticalFresh", scene, str(packet_id))
        if panel == "rayleigh_tdl":
            rng = np.random.default_rng(stable_seed(SEED, panel, str(packet_id)))
            gains = np.sqrt(np.array([.65, .25, .10]) / 2) * (rng.normal(size=3) + 1j*rng.normal(size=3))
            past = np.pad(x, (7, 0), mode="reflect")
            y = sum(g * past[7-d:7-d+len(x)] for g, d in zip(gains, (0, 2, 7)))
            std = np.sqrt((np.mean(abs(y)**2) * 10**(-14/10)) / 2)
            y += std * (rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))
        else:
            y, _ = ResidualChannel(cfg, seed, rx).process(x)
        result[i] = np.stack((y.real, y.imag))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite stress observations")
    return result


def model_routes(g, device):
    from experiments.pamf_r65_four_route.stage_polyphase import _stages
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    return {
        "raw": lambda x: x,
        "pilot_fir_mmse": PilotEqualizer(device),
        "coarse": g.frozen.coarse,
        "one_stage": lambda x: _stages(g.prior[:1], x, g.estimate_context(x))[0][0],
        "dura": lambda x, engine=InferenceEngine(g, reuse=True, graph=True, native=False): engine(x)[0],
    }


def batch_one_timing(x, reader, methods):
    samples = {}
    for name, restore in methods.items():
        def pipeline():
            return reader.classify_features(reader.features(restore(x)))
        for _ in range(20):
            pipeline()
        times = []
        for _ in range(60):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pipeline()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        samples[name] = dict(median_ms=float(np.median(times)), p95_ms=float(np.percentile(times,95)))
    return samples


def population(views_root, classes, task):
    m, ids, positions = handle(views_root, classes)
    selected = np.sort(np.random.default_rng(SEED + TASKS.index(task)).choice(len(ids), COUNT, replace=False))
    return m, ids[selected], selected if positions is None else positions[selected]


def predict(c):
    reject_truth(c)
    from experiments.pamf_paper_completion.runner import activate
    activate(c)
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.pamf_paper_completion.model import load_reader
    from experiments.pamf_channel_pl.restored_source import snapshot, unchanged
    r = Path(c["output_root"])
    r.mkdir(exist_ok=False)
    contract = read(c["source_contract"])
    device = "cuda:0"
    reader, _ = load_reader(c["source_output"], contract, dict(structure="full", signature_views=3,
        permuted=True, learning_regime="ssl", model_seed=2026092701), device)
    g = load_G(c["g"], contract, device)
    methods = model_routes(g, device)
    before = [snapshot(module) for module in (g,reader)]
    ref = np.load(c["source_smoke"], mmap_mode="r", allow_pickle=False)[:4]
    with torch.inference_mode():
        x = tensor(ref.astype(np.float32), device)
        for name, restore in methods.items():
            restored = restore(x)
            logits = reader.classify_features(reader.features(restored))
            if restored.shape != x.shape or logits.shape != (4,6) or not bool(logits.isfinite().all()):
                raise FloatingPointError(f"No-query smoke failed for {name}")
        timing = batch_one_timing(x[:1].contiguous(), reader, methods)
    write(r/"smoke.json", dict(status="PASS", query_reads=0, source_packets=4, timing_shared_gpu=timing))
    print("DURA_SOURCE_SMOKE_PASS", flush=True)
    fidelity = []
    for task in TASKS:
        m, ids, pos = population(c["views"][task], contract["classes"], task)
        clean = np.asarray(np.load(m["clean_ref"], mmap_mode="r", allow_pickle=False)[pos], dtype=np.float32)
        values = {}
        for panel in PANELS:
            if panel == "clean":
                observed = clean
            elif panel in VIEWS:
                observed = np.load(Path(m["observations_root"]) / f"{panel}.npy", mmap_mode="r", allow_pickle=False)[pos]
            else:
                observed = simulated(clean, ids, panel)
            buffers = {name: [] for name in METHODS}
            sums = {name: np.zeros(3, dtype=np.float64) for name in METHODS}
            with torch.inference_mode():
                for i in range(0,len(ids),128):
                    y = tensor(np.asarray(observed[i:i+128], dtype=np.float32), device)
                    ref = tensor(clean[i:i+128], device)
                    ref_feature = reader.features(ref)
                    energy = ref.square().mean((1,2)).clamp_min(1e-6)
                    ref_spec = torch.log1p(torch.fft.fft(torch.complex(ref[:,0],ref[:,1]),norm="ortho").abs().square())
                    for name, restore in methods.items():
                        z = restore(y)
                        features = reader.features(z)
                        logits = reader.classify_features(features)
                        if logits.shape != (len(y),6) or not bool(logits.isfinite().all()):
                            raise FloatingPointError("Invalid default-head output")
                        spec = torch.log1p(torch.fft.fft(torch.complex(z[:,0],z[:,1]),norm="ortho").abs().square())
                        buffers[name].extend(logits.argmax(1).tolist())
                        sums[name] += np.asarray([
                            ((z-ref).square().mean((1,2))/energy).sum().item(),
                            (spec-ref_spec).square().mean(1).sum().item(),
                            (1-F.cosine_similarity(features,ref_feature)).sum().item()])
            for name in METHODS:
                values[f"{name}__{panel}"] = np.asarray(buffers[name],dtype=np.int64)
                avg = sums[name] / len(ids)
                if not np.isfinite(avg).all():
                    raise FloatingPointError("Invalid paired fidelity metrics")
                fidelity.append(dict(task=task, panel=panel, route=name, packets=len(ids),
                    iq_nmse=float(avg[0]), log_power_spectrum_mse=float(avg[1]),
                    feature_cosine_distance=float(avg[2])))
            print("DURA_PANEL_PREDICTED",task,panel,len(ids),flush=True)
        np.savez(r/f"{task}.npz", ids=ids, **values)
    if any(not unchanged(module, state) for module,state in zip((g,reader),before)):
        raise ValueError("Frozen state changed")
    write_csv(r/"fidelity.csv",fidelity)
    write(r/"complete.json",dict(status="ALL_PREDICTIONS_COMPLETE", truth_read=False, query_fit=False,
                                 count_per_task=COUNT, panels=list(PANELS), routes=list(METHODS)))


def score(c):
    p = Path(c["prediction_output"])
    done = read(p/"complete.json")
    if done != dict(status="ALL_PREDICTIONS_COMPLETE", truth_read=False, query_fit=False,
                    count_per_task=COUNT, panels=list(PANELS), routes=list(METHODS)):
        raise ValueError("Truth stays closed without all preregistered predictions")
    predictions = {}
    for task in TASKS:
        _, expected, _ = population(c["tests"][task]["views_root"],c["classes"],task)
        with np.load(p/f"{task}.npz",allow_pickle=False) as a:
            if set(a.files) != {"ids"} | {f"{route}__{panel}" for route in METHODS for panel in PANELS}:
                raise ValueError("Missing frozen prediction cell")
            if not np.array_equal(a["ids"],expected):
                raise ValueError("Wrong sample population")
            values = {key:a[key] for key in a.files if key!="ids"}
            if any(x.shape!=(COUNT,) or x.dtype.kind not in "iu" or x.min()<0 or x.max()>=6 for x in values.values()):
                raise ValueError("Invalid frozen decisions")
            predictions[task] = expected, values

    rows = []
    for task,(ids,values) in predictions.items():
        truth = read(c["tests"][task]["truth"])
        labels = np.asarray([truth[i]["label"] for i in ids],dtype=np.int64)
        rx = np.asarray([truth[i]["receiver"] for i in ids])
        strata = [("overall","ALL",np.ones(COUNT,dtype=bool))]
        strata.extend(("transmitter",name,labels==i) for i,name in enumerate(c["classes"]))
        strata.extend(("receiver",str(i),rx==i) for i in np.unique(rx))
        for panel in PANELS:
            ref = values[f"raw__{panel}"]
            for route in METHODS:
                result = values[f"{route}__{panel}"]
                for dim, name, use in strata:
                    a,b = ref[use]==labels[use],result[use]==labels[use]
                    stat = metrics(labels[use],result[use],6)
                    stat.pop("confusion")
                    rows.append(dict(task=task,panel=panel,route=route,dimension=dim,stratum=name,
                        repairs=int((~a & b).sum()),damage=int((a & ~b).sum()),
                        net=int(b.sum()-a.sum()),**stat))
    root=Path(c["output_root"])
    root.mkdir(exist_ok=False)
    write_csv(root/"scores.csv",rows)
    write(root/"complete.json",dict(status="SCORED_COMPLETE",truth_last=True,score_rows=len(rows)))
    print("DURA_ADVISOR_SCORED",len(rows),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("predict","score"))
    p.add_argument("--config",required=True)
    args=p.parse_args()
    globals()[args.action](read(args.config))
