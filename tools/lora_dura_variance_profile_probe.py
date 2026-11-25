"""Compare received local variance with time-shifted and original DURA controls."""

import argparse
import json
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_capacity_source import RECIPES, RUN, recover
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.lapped_posterior import analysis
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.spectral_block_source import load_models
from tools.lora_robust_observation_probe import FIELDS, ORIGIN_CODE, fidelity


WINDOW, HOP = 64, 32
ARMS = ('direct', 'expanded_half', 'local_variance', 'shifted_variance')
CONDITIONS = ('clean', *((kind, dose) for dose in (.18, .38)
                        for kind in ('white', 'colored', 'burst', 'uneven_chirps')))


def short_time_bands(y):
    device, dtype = y.device, y.real.dtype
    window = ((torch.arange(WINDOW, device=device, dtype=dtype)+.5)*torch.pi/WINDOW).sin()
    frequency = torch.fft.fftshift(torch.fft.fftfreq(WINDOW, device=device, dtype=dtype))
    positive = (frequency >= .25) & (frequency <= .4375)
    negative = (frequency <= -.25) & (frequency >= -.4375)
    energy = analysis(y, window).abs().square()
    return torch.stack([energy[..., band].mean(-1) for band in (positive, negative)], dim=-1)


def relative_log(power):
    logged = power.clamp_min(1e-10).log()
    return logged-logged.mean(-2, keepdim=True)


def received_profile(y):
    sided = relative_log(short_time_bands(y))
    first, second = sided.unbind(-1)
    agree = (first*second).sum(-1) / (first.square().sum(-1)*second.square().sum(-1)).clamp_min(1e-10).sqrt()
    change = (.75*agree.clamp(0., 1.)[..., None] * sided.mean(-1)).clamp(-log(2.), log(2.))
    fraction = torch.arange(HOP, device=y.device, dtype=y.real.dtype)/HOP
    weights = (change[..., None]*(1.-fraction) + change.roll(-1, dims=-1)[..., None]*fraction)
    return weights.flatten(-2).exp(), agree, change


def inject(reference, condition):
    if condition == 'clean':
        return reference
    kind, dose = condition
    signal = complex_iq(reference)
    rng = torch.Generator(device=signal.device).manual_seed(2026112593)
    noise = torch.randn(signal.shape, dtype=signal.dtype, device=signal.device, generator=rng)
    if kind == 'colored':
        noise = noise + .85*noise.roll(1, dims=-1)
    elif kind == 'burst':
        start = torch.randint(256, signal.shape[:2]+(1,), device=signal.device, generator=rng)
        duration = torch.randint(48, 97, start.shape, device=signal.device, generator=rng)
        phase = (torch.arange(256, device=signal.device)-start)%256
        noise = noise*(.2+2.8*(phase<duration))
    elif kind == 'uneven_chirps':
        strength = torch.rand(signal.shape[:2]+(1,), device=signal.device, generator=rng)
        noise = noise*(.2+2.8*strength.square())
    scale = noise.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-10).sqrt()
    return real_iq(signal+dose*noise/scale)


def summarize(values):
    result = dict(rows=len(values), medians=np.median(values, axis=0).tolist())
    for arm, index in zip(ARMS[2:], (2, 3), strict=True):
        base, proposed = values[:, 1, :3], values[:, index, :3]
        percent = 100.*(base-proposed)/np.maximum(base, 1e-9)
        result[arm] = dict(reduction=np.median(percent, 0).tolist(), improved=(proposed<base).sum(0).tolist())
    paired = values[:, 2, :3] < values[:, 3, :3]
    result['local_beats_shifted'] = paired.sum(0).tolist()
    return result


@torch.no_grad()
def measure(rt, device):
    started = perf_counter()
    if (rt['run_id'], rt['code_commit']) != (RUN, ORIGIN_CODE):
        raise ValueError('Expected the original expanded DURA U-only model ancestry')
    numerical_policy()
    source = Path(rt['source_root'])
    contract = source_contract(source)
    native, rx, _, _ = source_role(source, contract, 'U_s')
    _, _, held = partition(rx, fit=4000)
    ground = tensor(np.stack([chirp_windows(native[k]) for k in held]), device)
    tracker = PacketRestorer('consensus').to(device).eval()
    models, threshold = load_models(Path(rt['run_root']), contract, device, {'expanded': RECIPES['expanded']})
    model, = models
    frequency = (256*torch.fft.fftfreq(256, device=device)).abs()
    band = (frequency >= 9) & (frequency <= 32)
    groups = {'all': np.ones(len(held), dtype=bool), **{key: rx[held] == key for key in SOURCE}}
    result = {}
    for condition in CONDITIONS:
        observed = inject(ground, condition)
        rows, profiles, active_count, unchanged_count = [], [], 0, 0
        for clean, iq in zip(ground.split(16), observed.split(16), strict=True):
            y, phase, q, rms = coordinates(tracker, iq)
            frame = phase*rms
            profile, agreement, local_change = received_profile(y)
            active = corruption_score(y) > threshold
            original, count, _ = recover(iq, tracker, models, threshold)
            if int(active.sum()) != count:
                raise ValueError('Local solve and original DURA must share the same receive-only gate')
            active_count += count
            unchanged_profile = (profile == 1.).all((-2, -1))
            unchanged_count += int((active & unchanged_profile).sum())
            waves = [iq, original[3]]
            for field in (profile, profile.roll(y.shape[-1]//2, dims=-1)):
                output = iq.clone()
                if count:
                    stages, *_ = model.stages(y[active], q[active], rms[active], solve_profile=field[active])
                    edit = .5*(stages[-1]-y[active])
                    output[active] += real_iq(edit*phase[active]*rms[active])
                if not torch.equal(output[unchanged_profile], waves[1][unchanged_profile]):
                    raise ValueError('A unit profile must preserve the archived IQ readout exactly')
                waves.append(output)
            rows.extend(torch.stack([fidelity(clean, wave, frame, band) for wave in waves], dim=1).tolist())
            if condition != 'clean':
                epsilon = complex_iq(iq-clean)/frame
                target = relative_log(short_time_bands(epsilon).mean(-1, keepdim=True)).squeeze(-1)
                amp = 10./log(10.)
                error = torch.stack((target, local_change-target, local_change.roll(4, dims=-1)-target))
                rms_db = amp*error.square().mean((-2, -1)).sqrt().T
                profiles.extend(torch.cat((agreement.mean(1, keepdim=True), rms_db), -1).tolist())
        rows = np.asarray(rows)
        if not np.isfinite(rows).all() or (profiles and not np.isfinite(profiles).all()):
            raise FloatingPointError('Nonfinite local precision probe')
        if condition == 'clean' and (active_count or np.any(rows != rows[:, :1])):
            raise ValueError('Clean held U must retain the original bypass')
        name = condition if isinstance(condition, str) else f'{condition[0]}_{condition[1]:.2f}'
        result[name] = dict(active=active_count, active_unit_profile=unchanged_count,
                            groups={label: summarize(rows[mask]) for label, mask in groups.items()})
        if profiles:
            result[name]['profile_medians'] = np.median(profiles, axis=0).tolist()
    return dict(status='VERIFIED', role='U_s', source_commit=rt['code_commit'], held=len(held),
                fields=FIELDS, arms=ARMS, profile_fields=('side_agreement', 'pooled_error_db',
                'local_error_db', 'shifted_error_db'), conditions=result, seconds=perf_counter()-started,
                optimizer_updates=0, added_parameters=0, TX_reads=0, V_reads=0, F_reads=0, target_reads=0, trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(measure(read(args.runtime), args.device), allow_nan=False))
