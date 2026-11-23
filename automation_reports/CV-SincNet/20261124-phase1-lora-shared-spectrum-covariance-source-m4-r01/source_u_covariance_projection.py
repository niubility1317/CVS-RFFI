from pathlib import Path
import json
import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.coupled_echo_source import RUN
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.runner import activate

base=Path('/home/szu2070436088/2510044040/CV-SincNet')
rt=read(base/'releases'/f'{RUN}-release'/'runtime.json')
activate(rt)
torch.set_num_threads(2)
root=Path(rt['source_root'])
raw,rx,ids,_=source_role(root,source_contract(root),'U_s')
_,_,held=partition(rx)
clean=tensor(np.stack([chirp_windows(raw[i]) for i in held]),'cpu')
tracker=PacketRestorer('consensus').eval()
with torch.no_grad():
    canonical=coordinates(tracker,clean)[0].mean(1).to(torch.complex128)
t=torch.linspace(-1,1,canonical.shape[-1],dtype=torch.float64)
basis=torch.linalg.qr(torch.stack((torch.ones_like(t),t,t.square()),dim=1)).Q.to(canonical.dtype)
deformations={'direct':canonical,'without_linear':canonical-(canonical@basis[:,1:2])@basis[:,1:2].T,
              'without_linear_and_quadratic':canonical-(canonical@basis[:,1:])@basis[:,1:].T}
freq=(256*torch.fft.fftfreq(256)).round().abs()
wanted=(freq>=1)&(freq<=8)
anchored=(freq>=9)&(freq<=64)
def real(z):
    return torch.cat((z.real,z.imag),dim=-1)
records={}
for method,iq in deformations.items():
    spec=torch.fft.fft(iq,norm='ortho')
    features=real(spec[...,anchored])
    targets=real(spec[...,wanted])
    rows={}
    for site in SOURCE:
        choose=np.flatnonzero(rx[held]==site).tolist()
        fit=np.flatnonzero(rx[held]!=site).tolist()
        xm,ym=features[fit].mean(0),targets[fit].mean(0)
        x=features[fit]-xm
        y=targets[fit]-ym
        kernel=x.T@x
        model=torch.linalg.solve(kernel+.03*kernel.diagonal().mean()*torch.eye(len(kernel)),x.T@y)
        predicted=(features[choose]-xm)@model+ym
        actual=targets[choose]
        centered=actual-actual.mean(0)
        predicted_centered=predicted-predicted.mean(0)
        denom=centered.square().sum()
        shift=predicted_centered.roll(47,0)
        rows[site]=dict(
             leave_rx_r2=float(1-(predicted-actual).square().sum()/(actual-ym).square().sum()),
             within_rx_centered_r2=float(1-(predicted_centered-centered).square().sum()/denom),
             shuffled_pair_centered_r2=float(1-(shift-centered).square().sum()/denom),
             centered_power=float(denom/len(actual)))
    records[method]=rows
print(json.dumps(dict(packets=len(held),fold_train=400,source_role='held_U_s',
                      F_reads=0,V_reads=0,TX_reads=0,target_reads=0,rows=records)),flush=True)
