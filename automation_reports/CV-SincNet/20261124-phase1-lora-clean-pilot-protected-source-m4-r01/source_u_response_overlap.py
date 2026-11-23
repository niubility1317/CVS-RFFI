import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.clean_pilot_response import CleanPilotResponse, complex_vector, relative_response
from experiments.pamf_lora_main.clean_pilot_response_source import RUN, calibrate
from experiments.pamf_lora_main.crossfit_pilot import LAGS, estimate_shared, generic_taps
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

torch.set_num_threads(2)
project = Path('/home/szu2070436088/2510044040/CV-SincNet')
rt = read(project/'releases'/f'{RUN}-release'/'runtime.json')
(source, _, held), _ = setup(rt, 'cpu')
tracker = PacketRestorer('consensus').eval()
calibration = CleanPilotResponse()

class BufferHolder:
    pass

holder = BufferHolder()
holder.calibration = calibration
calibrate(holder, tracker, source)
frequency = torch.fft.fftfreq(256)
phase = -2 * torch.pi * torch.tensor(LAGS)[:, None] * frequency
transfer = torch.polar(torch.ones_like(phase), phase)
band = frequency.abs() <= .25
rng = torch.Generator().manual_seed(2026112452)
levels = torch.tensor([.1, .5, .9])

def energy_ratio(vector):
    filtered = vector @ calibration.shrink
    n = band.numel()
    return {name: (filtered[..., columns].square().sum(-1) /
                   vector[..., columns].square().sum(-1).clamp_min(1e-16))
            for name, columns in (('whole',slice(None)), ('occupied',torch.cat((band,band))))}

def quantile_rows(columns):
    return {k: v.quantile(levels).tolist() for k,v in columns.items()}

with torch.no_grad():
    original = []
    for wave in held.split(20):
        y, rotation, quality, _ = coordinates(tracker, wave)
        raw, valid = estimate_shared(y, rotation, quality, observed_noise_rms(y).square())
        original.append((relative_response(raw).squeeze(1) - 1, valid))
    raw = torch.cat([x for x,_ in original])
    support = torch.cat([x for _,x in original])
    clean = complex_vector(raw[support > 0]) - calibration.mean
    truth = relative_response(1 + generic_taps(complex_iq(held), rng) @ transfer) - 1
    modes = torch.cat((transfer, 1j*transfer))*.1
    tap_response = complex_vector(relative_response(1+modes)-1)
    results = dict(fit=500, held=len(held), clean_supported=int((support > 0).sum()), target_reads=0,
                   F_reads=0,V_reads=0,TX_reads=0, quantiles=levels.tolist(),
                   retained_clean_response_energy=quantile_rows(energy_ratio(clean)),
                   retained_generic_response_energy=quantile_rows(energy_ratio(complex_vector(truth))),
                   retained_tap_energy=energy_ratio(tap_response)['occupied'].reshape(2,-1).tolist(),
                   lags=LAGS, mode_order=['real','imaginary'])
print(json.dumps(results), flush=True)
