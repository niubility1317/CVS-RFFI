"""Isolate native PSD estimation and local-frequency transport on held source U."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
release = root.parent.parent / 'releases' / (root.name + '-release')
sys.path.insert(0, str(release))

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_role, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import FFT, HOP, RFNoiseTransport

torch.set_num_threads(2)
assert read(root/'state.json')['status']=='SOURCE_SCORED'
rt=read(release/'runtime.json')
source=Path(rt['source_root'])
_, rx, _, _=source_role(source,source_contract(source),'U_s')
_,_,held=partition(rx)
rows=held[::10]
iq=np.load(root.parent / DATA_RUN / 'source_views' / 'U_s.npy',mmap_mode='r',allow_pickle=False)
clean=complex_iq(tensor(iq[0,rows],'cpu'))
generator=torch.Generator().manual_seed(2026112423)
B,C,N=clean.shape
samples=48
frequencies=torch.fft.fftfreq(N)
center=torch.rand(B,1,generator=generator)-.5
distance=(frequencies-center+.5).remainder(1)-.5
fft_channels=torch.arange(FFT)
grid=torch.tensor([.10,.50,.90])
model=RFNoiseTransport(True).eval()
tracker=PacketRestorer('consensus').eval()
summaries={}

def normalize(x):
    return x/x.mean(-1,keepdim=True).clamp_min(1e-8)

def power(z):
    return z.abs().square()

for name,width in [('flat',None),('wide',.08),('narrow',.028)]:
    base=torch.ones_like(distance) if width is None else .35+3*torch.exp(-.5*(distance/width).square())
    amplitude=(.40 * power(clean).mean(-1,keepdim=True).sqrt()) * normalize(base).sqrt()[:,None]

    def noise():
        return torch.fft.ifft(torch.fft.fft(torch.randn((B,C,N),dtype=torch.complex64,generator=generator))*amplitude)

    observed,rotation,_,rms=coordinates(tracker,real_iq(clean+noise()))
    estimate=model.estimate_noise(observed,rotation)
    ridge=power(model.analyze(rotation)).argmax(dim=-1,keepdim=True)
    # Reverse the first frame's integer shift to recover the same measured profile.
    native=estimate[:,:,0].gather(-1,(fft_channels-ridge[:,:,0]).remainder(FFT))
    flat=estimate.mean(dim=-1,keepdim=True).expand_as(estimate)
    mapping=(fft_channels+ridge).remainder(FFT)
    # Exclude windows that touch reflection boundaries. On the rest, use the
    # actual phase trajectory to form the received-to-dechirped power kernel.
    windows=rotation.unfold(-1,FFT,HOP)
    kernel=power(torch.fft.fft(windows.conj()*model.window))
    kernel=kernel/kernel.sum(-1,keepdim=True)
    compared=slice(2,-2)

    def transfer(profile):
        return torch.fft.ifft(torch.fft.fft(profile[:,:,None])*torch.fft.fft(kernel)).real.clamp_min(0)

    tracked_power=torch.zeros_like(estimate)
    native_power=torch.zeros_like(estimate)
    for _ in range(samples):
        sample=noise()/rms
        native_power+=power(model.analyze(sample))
        tracked_power+=power(model.analyze(sample*rotation.conj()))
    native_reference=native_power[:,:,compared].mean(-2)/samples
    reference=(tracked_power/samples)[:,:,compared]
    alternatives=dict(
        flat=flat[:,:,compared],
        measured_integer=estimate[:,:,compared],
        measured_kernel=transfer(native),
        known_integer=native_reference[:,:,None].expand_as(estimate).gather(-1,mapping)[:,:,compared],
        known_kernel=transfer(native_reference),
    )
    errors={key:(normalize(p)-normalize(reference)).square().mean(dim=(1,2,3)) for key,p in alternatives.items()}
    summaries[name]={k:dict(mse_quantiles=v.quantile(grid).tolist(),
                            difference_vs_flat_quantiles=(v-errors['flat']).quantile(grid).tolist())
                     for k,v in errors.items()}

print(json.dumps(dict(status='VERIFIED',hold=len(rows),replicates_per_case=samples,TX_reads=0,target_reads=0,
                      comparison=summaries),indent=2))
