"""Check a LoRa relative echo estimator on held source U and generic disturbances."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import clean_threshold, partition
from experiments.pamf_lora_main.cross_frequency import CrossFrequencyPrior
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.relative_echo import SHORT, action, correction
from experiments.pamf_lora_main.relative_echo_source import observe


REGIMES = ("ground", "gaussian", "varying_echo", "common_echo", "varying_echo_and_gaussian")
MODES = ("direct", "scalar", "short")
EXTENDED = ("short_evidence_gate", "short_evidence_gate_private")
SEED = 2026112477


def generic(source, regime, generator):
    if regime in ("varying_echo", "common_echo", "varying_echo_and_gaussian"):
        coefficient = torch.randn((len(source), 6, len(SHORT)), device=source.device,
                                  dtype=source.dtype, generator=generator)
        coefficient = .15 * coefficient
        coefficient[..., 0] *= .5
        if regime == "common_echo":
            coefficient = coefficient[:, :1].expand_as(coefficient)
        else:
            coefficient -= coefficient.mean(1, keepdim=True)
        source = action(source, torch.ones_like(source), coefficient, SHORT)
    if regime in ("gaussian", "varying_echo_and_gaussian"):
        standard = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        source = source + .15 * standard
    return source


def wave_error(wave, physical, frame):
    residual = torch.fft.fft((wave - physical) / frame, norm="ortho")
    reference = torch.fft.fft(physical / frame, norm="ortho")
    distance = torch.fft.fftfreq(wave.shape[-1], device=wave.device).abs()
    bandlist = ((distance > 0) & (distance <= 1/32), (distance > 1/32) & (distance <= 1/8))
    output = [(wave - physical).abs().square().mean((1, 2))]
    for value, original in ((residual.mean(1, keepdim=True), reference.mean(1, keepdim=True)),
                            (residual - residual.mean(1, keepdim=True), reference - reference.mean(1, keepdim=True))):
        for band in bandlist:
            power = original[..., band].abs().square().mean((1, 2))
            output.append(value[..., band].abs().square().mean((1, 2)) / power.clamp_min(1e-5))
    return torch.stack(output, dim=1)


def measure(physical, delivered, tracker, noise, threshold, echo_threshold):
    y, phase, rms, (scalar, short, evidence) = observe(real_iq(delivered), tracker, noise)
    if not torch.allclose(scalar.mean(1), torch.zeros_like(scalar[:, 0]), atol=1e-6):
        raise ValueError("Scalar reference gauge changed")
    if not torch.allclose(short.mean(1), torch.zeros_like(short[:, 0]), atol=1e-6):
        raise ValueError("Short relative reference gauge changed")
    active = corruption_score(y) > threshold
    frame = phase * rms
    scalar_edit, short_edit = (correction(y, phase, coeff) for coeff in (scalar, short))
    edits = [edit * frame * active[:, None, None] for edit in (scalar_edit, short_edit)]
    echo_active = active
    if echo_threshold is not None:
        echo_active = active | (evidence[4].flatten() > echo_threshold)
        for edit in (short_edit, short_edit - short_edit.mean(1, keepdim=True)):
            edits.append(edit * frame * echo_active[:, None, None])
    risks = torch.stack([wave_error(wave, physical, frame) for wave in (delivered, *(delivered + e for e in edits))],
                        dim=1)
    held = torch.stack([value.mean(1) for value in evidence[:3]], dim=1)
    return (risks.cpu().numpy(), held.cpu().numpy(), (evidence[3].flatten() > 0).cpu().numpy(),
            active.cpu().numpy(), echo_active.cpu().numpy())


@torch.inference_mode()
def probe(source, device, *, extended):
    started = perf_counter()
    contract = source_contract(source)
    iq, receivers, _, _ = source_role(source, contract, "U_s")
    _, calibration, evaluation = partition(receivers, fit=4000, held=200 if extended else 100)
    tracker = PacketRestorer("consensus").to(device).eval()
    prior = CrossFrequencyPrior().to(device)

    def wave(indices):
        return torch.from_numpy(np.stack([chirp_windows(iq[i]) for i in indices])).to(device)

    threshold = clean_threshold(tracker, wave(calibration))
    echo_threshold = None
    if extended:
        evaluation = evaluation.reshape(len(SOURCE), 200)[:, 100:].ravel()
        echo_threshold = 1.05 * torch.cat([
            observe(values, tracker, prior)[3][2][4]
            for values in wave(calibration).split(48)
        ]).max()
    result = {}
    for regime in REGIMES:
        generator = torch.Generator(device=device).manual_seed(SEED)
        results = [measure(ground, generic(ground, regime, generator), tracker, prior, threshold, echo_threshold)
                   for ground in (complex_iq(wave(group)) for group in np.array_split(evaluation, 20))]
        risks, held, evidence, active, echo_active = [np.concatenate(arm) for arm in zip(*results, strict=True)]
        if not (np.isfinite(risks).all() and np.isfinite(held).all()):
            raise ValueError("Invalid relative echo measurement")
        result[regime] = {}
        for receiver in SOURCE:
            rows = receivers[evaluation] == receiver
            result[regime][receiver] = dict(n=int(rows.sum()), active=int(active[rows].sum()),
                extra_support=int((evidence & active)[rows].sum()), echo_active=int(echo_active[rows].sum()),
                prediction_mse=held[rows].mean(0).tolist(), ground_error=risks[rows].mean(0).tolist())
    return dict(status="SOURCE_U_ONLY", seed=SEED, calibration=len(calibration), held=len(evaluation),
                modes=(*MODES, *EXTENDED) if extended else MODES,
                echo_threshold=float(echo_threshold) if extended else None,
                physical_columns=("whole", "common_1_8", "common_9_32", "private_1_8", "private_9_32"),
                prediction_columns=("identity", "scalar", "short"), results=result,
                target_reads=0, F_reads=0, TX_reads=0, V_reads=0, worker_seconds=perf_counter()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extended", action="store_true")
    options = parser.parse_args()
    result = probe(options.source, options.device, extended=options.extended)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(options.output)
