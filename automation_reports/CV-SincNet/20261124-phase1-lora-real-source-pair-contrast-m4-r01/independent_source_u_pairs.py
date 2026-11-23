
import json
import sys
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer,complex_iq,real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.pilot_joint import apply_echo,echo_taps
from experiments.pamf_lora_main.real_pair import BANDS,signed_difference
from experiments.pamf_lora_main.real_pair_source import RECIPES,RUN
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

torch.set_num_threads(2)
strength = float(sys.argv[1]) if len(sys.argv) > 1 else 1.
device='cpu'
parent=Path('/home/szu2070436088/2510044040/CV-SincNet')
rt=read(parent/'releases'/f'{RUN}-release'/'runtime.json')
root=Path(rt['run_root'])
(_,_,held),contract=setup(rt,device)
models={name:load_model(root,recipe,contract,device) for name,recipe in RECIPES.items()}
thresholds={value[1] for value in models.values()}
assert len(thresholds)==1
threshold=thresholds.pop()
tracker=PacketRestorer('consensus').eval()
rng=torch.Generator().manual_seed(2026112433)
arms=('direct','point','point_half','pair','pair_half')
bins={name:[] for name in arms}
groups=[]
with torch.no_grad():
    for source in held.chunk(len(SOURCE)):
        for iq in source.split(12):
            clean=complex_iq(iq)
            # Every distinct packet gets its own generic echo and noise in this check.
            draw=colored_observation(apply_echo(clean,echo_taps(clean,rng)),rng)
            distorted=real_iq(clean + strength * (draw-clean) if strength != 1. else draw)
            y,rot,q,rms=coordinates(tracker,distorted)
            _,true_rot,_,true_rms=coordinates(tracker,iq)
            true_basis=true_rot*true_rms
            ref=torch.fft.fft(signed_difference(clean/true_basis),norm='ortho')
            active=(corruption_score(y)>threshold)[...,None,None]
            s=observed_noise_rms(y).squeeze(-1).unflatten(0,(-1,2)).mean((1,2))
            groups.append(s)
            waves={'direct':y}
            for name,(model,_) in models.items():
                fine=model(y,q,rot)[-1]
                waves[name]=torch.where(active,fine,y)
                waves[name+'_half']=torch.where(active,torch.lerp(y,fine,.5),y)
            freq=torch.fft.fftfreq(ref.shape[-1]).abs()*ref.shape[-1]
            for name in arms:
                restored=torch.fft.fft(signed_difference(waves[name]*rot*rms/true_basis),norm='ortho')
                result=[]
                for lo,hi in BANDS:
                    freqs=(freq>=lo)&(freq<=hi)
                    signal=ref[...,freqs]
                    candidate=restored[...,freqs]
                    reference_power=signal.abs().square().mean(dim=(1,2))
                    error=(candidate-signal).abs().square().mean(dim=(1,2))
                    gain=(candidate*signal.conj()).mean(dim=(1,2))/reference_power.clamp_min(1e-10)
                    result.append(torch.stack(((error/(.002+reference_power)).log1p(),gain.real,gain.imag),dim=-1))
                bins[name].append(torch.stack(result,dim=1))
s=torch.cat(groups)
observations={mode:torch.cat(values) for mode,values in bins.items()}
outputs={}
for label,mask in (('all',s>=0),('up_to_0.15',s<=.15),('0.15_to_0.40',(s>.15)&(s<=.40)),('above_0.40',s>.40)):
    outputs[label]=dict(physical_pairs=int(mask.sum()),
                        modes={mode:value[mask].mean(0).tolist() for mode,value in observations.items()} if mask.any() else {})
print(json.dumps(dict(held_packets=len(held),strength=strength,distinct_draws_per_packet=True,target_reads=0,F_reads=0,
                      V_reads=0,TX_reads=0,seed=2026112433,bands_khz=BANDS,
                      measurement_order=('log1p_floored_complex_error','real_gain','imag_gain'),
                      output=outputs)),flush=True)
