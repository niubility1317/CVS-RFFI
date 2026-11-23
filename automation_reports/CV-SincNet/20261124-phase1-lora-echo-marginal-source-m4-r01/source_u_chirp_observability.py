import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.crossfit_pilot import LAGS, apply_transfer, generic_taps
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.echo_marginal import common_weights
from experiments.pamf_lora_main.echo_marginal_source import RUN
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import colored_observation

torch.set_num_threads(2)
project = Path('/home/szu2070436088/2510044040/CV-SincNet')
rt = read(project/'releases'/f'{RUN}-release'/'runtime.json')
(_, _, held), _ = setup(rt, 'cpu')
tracker = PacketRestorer('consensus').eval()
rng = torch.Generator().manual_seed(2026112455)
frequency = (256*torch.fft.fftfreq(256)).round().abs()
bins = (frequency>0) & (frequency<=32)
low = frequency[bins] <= 8
quantiles = torch.tensor([.1,.5,.9])
results = {}

def contrast_ratio(columns, weight):
    common = (columns * weight[...,None]).sum(1,keepdim=True)
    denom = (columns.abs().square() * weight[...,None]).sum((1,2,3))
    contrast = ((columns-common).abs().square() * weight[...,None]).sum((1,2,3))
    return contrast / denom.clamp_min(1e-8)

with torch.no_grad():
    for iq in held.split(20):
        clean = complex_iq(iq)
        acted = apply_transfer(clean, generic_taps(clean,rng))
        disturbance = colored_observation(acted,rng)-acted
        for stage,wave in (('clean',clean),('light_echo',acted+.25*disturbance),('broad_echo',acted+.85*disturbance)):
            y,template,q,_ = coordinates(tracker,real_iq(wave))
            weight = common_weights(q)
            gain = y.mean(-1,keepdim=True)
            relative = torch.stack([template.roll(lag,-1)*template.conj() for lag in LAGS],dim=-1)
            relative = torch.fft.fft(relative,dim=2,norm='ortho')[:,:,bins]
            for band,selected in (('1_to_8',low),('9_to_32',~low)):
                shape = relative[:,:,selected]
                for coordinate, columns in (('gain_normalized',shape),('measured_gain',shape*gain[...,None])):
                    r = contrast_ratio(columns, weight)
                    results.setdefault(stage,{}).setdefault(band,{}).setdefault(coordinate,[]).append(r)
for stage in results.values():
    for band in stage.values():
        for c, rows in band.items():
            vector = torch.cat(rows)
            assert len(vector)==len(held)
            band[c] = vector.quantile(quantiles).tolist()
print(json.dumps(dict(packets=len(held),targets_read=0,F_reads=0,V_reads=0,TX_reads=0,
                      quantiles=[.1,.5,.9],results=results)),flush=True)
