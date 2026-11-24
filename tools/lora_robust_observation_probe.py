"""Compare quadratic and robust observation solves on held source U LoRa IQ."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_capacity_source import RECIPES, RUN, recover
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.spectral_block_source import load_models


ORIGIN_CODE = '3de50180aaaa1179587a753d4f6dde201fc989c6'
CONDITIONS = ('clean', *((shape, rms) for rms in (.18, .38) for shape in ('white', 'intermittent', 'colored')))
FIELDS = ('physical_nmse', 'private_complex_nmse_9_32', 'private_amplitude_nmse_9_32', 'private_real_coherence_9_32')


def inject(reference, condition, generator):
    if condition == 'clean':
        return reference
    kind, rms = condition
    signal = complex_iq(reference)
    noise = torch.randn(signal.shape, dtype=signal.dtype, device=signal.device, generator=generator)
    if kind == 'intermittent':
        blocks = torch.rand((*signal.shape[:-1], signal.shape[-1] // 4), device=signal.device, generator=generator)
        envelope = .30 + 4.5 * (blocks < .06).repeat_interleave(4, dim=-1)
        noise = noise * envelope
    elif kind == 'colored':
        noise = noise + .85 * noise.roll(1, dims=-1)
    energy = noise.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-10)
    return real_iq(signal + rms * noise / energy.sqrt())


def fidelity(reference, value, frame, band):
    target, updated = complex_iq(reference), complex_iq(value)
    absolute = (target-updated).abs().square().sum((1, 2)) / target.abs().square().sum((1, 2)).clamp_min(1e-8)

    def private(signal):
        tracked = signal / frame
        residual = tracked - tracked.mean(1, keepdim=True)
        return torch.fft.fft(residual, norm='ortho')[..., band]

    target, updated = private(target), private(updated)
    energy = target.abs().square().sum((1, 2)).clamp_min(1e-8)
    complex_error = (updated-target).abs().square().sum((1, 2)) / energy
    amplitude_error = (updated.abs()-target.abs()).square().sum((1, 2)) / energy
    real_coherence = (updated*target.conj()).real.sum((1, 2)) / (
        energy*updated.abs().square().sum((1, 2)).clamp_min(1e-8)).sqrt()
    return torch.stack((absolute, complex_error, amplitude_error, real_coherence), dim=-1)


def summary(values):
    base, robust = values[:, 1], values[:, 2]
    gain = 100. * (base[:, :3]-robust[:, :3]) / np.maximum(base[:, :3], 1e-9)
    return dict(n=len(values), median=np.median(values, axis=0).tolist(),
                relative_error_reduction=np.median(gain, axis=0).tolist(),
                improved=(robust[:, :3] < base[:, :3]).sum(0).tolist())


@torch.no_grad()
def measure(runtime, device):
    started = perf_counter()
    if (runtime['run_id'], runtime['code_commit']) != (RUN, ORIGIN_CODE):
        raise ValueError('Expected the original source-U-only 40,000-step capacity run')
    numerical_policy()
    source = Path(runtime['source_root'])
    contract = source_contract(source)
    native, rx, _, _ = source_role(source, contract, 'U_s')
    _, _, held = partition(rx, fit=4000)
    tracker = PacketRestorer('consensus').to(device).eval()
    models, threshold = load_models(Path(runtime['run_root']), contract, device, {'expanded': RECIPES['expanded']})
    ground = tensor(np.stack([chirp_windows(native[k]) for k in held]), device)
    freq = (256*torch.fft.fftfreq(256, device=device)).abs()
    band = (freq >= 9) & (freq <= 32)
    groups = {'all': np.ones(len(held), dtype=bool), **{name: rx[held] == name for name in SOURCE}}
    result = {}
    for condition in CONDITIONS:
        generator = torch.Generator(device=device).manual_seed(2026112612)
        observed = inject(ground, condition, generator)
        results, hits = [], 0
        for clean, received in zip(ground.split(16), observed.split(16), strict=True):
            _, phase, _, level = coordinates(tracker, received)
            frame = phase*level
            quadratic, active, _ = recover(received, tracker, models, threshold)
            robust, changed, _ = recover(received, tracker, models, threshold, robust=True)
            if changed != active:
                raise ValueError('The two solvers must use identical source-only gating')
            hits += active
            results.append(torch.stack([fidelity(clean, wave, frame, band)
                                        for wave in (received, quadratic[3], robust[3])], dim=1).tolist())
        samples = np.concatenate(results)
        if not np.isfinite(samples).all():
            raise FloatingPointError('Nonfinite paired source waveform measure')
        name = condition if isinstance(condition, str) else f'{condition[0]}_{condition[1]:.2f}'
        result[name] = dict(active=hits, groups={label: summary(samples[select]) for label, select in groups.items()})
    return dict(status='VERIFIED', input_role='U_s', held=len(held), fields=FIELDS,
                readouts=('direct', 'quadratic_half', 'robust_half'), samples_per_chirp=256,
                steps=0, added_parameters=0, target_reads=0, TX_reads=0, F_reads=0, V_reads=0, trace_reads=0,
                seconds=perf_counter()-started, conditions=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(measure(read(args.runtime), args.device), allow_nan=False), flush=True)
