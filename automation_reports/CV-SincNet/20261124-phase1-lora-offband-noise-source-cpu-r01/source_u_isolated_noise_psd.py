
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from experiments.cric_dura_comparison.io import read,tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer,complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract,source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise import conditional_noise
from experiments.pamf_lora_main.offband_noise_source import BANDS,band,measurements,replay
from experiments.pamf_lora_main.runner import activate


@torch.no_grad()
def main():
    rt=read(Path('runtime.json'))
    activate(rt)
    root=Path(rt['run_root'])
    if read(root/'state.json')['status']!='SOURCE_DIAGNOSTIC_COMPLETE':
        raise ValueError('Original fixed source-U probe must already be complete')
    source=Path(rt['source_root'])
    contract=source_contract(source)
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held=partition(rx)
    count=30
    selected=np.concatenate([block[:count] for block in np.split(held,len(SOURCE))])
    groups=torch.arange(len(SOURCE)).repeat_interleave(count)
    tracker=PacketRestorer('consensus').eval()
    report={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            all_views={}
            for scene in (VIEWS[3],VIEWS[-1]):
                items=((route,scene,raw[i],rx[i],ids[i]) for i in selected)
                iq,noise_iq=(tensor(np.stack(p),'cpu') for p in
                             zip(*pool.map(replay,items,chunksize=6),strict=True))
                y,rotation,_,rms=coordinates(tracker,iq)
                spectrum=torch.fft.fft(y,norm='ortho')
                real_noise=complex_iq(noise_iq)
                raw_power=torch.fft.fft(real_noise,norm='ortho').abs().square()
                raw_power=raw_power/raw_power.mean(-1,keepdim=True).clamp_min(1e-12)
                measured_profile=raw_power.mean((0,1))
                oracle=torch.empty_like(raw_power)
                for fold in range(len(SOURCE)):
                    chosen=groups==fold
                    # Exact noise is used only for this diagnostic, in other source RXs.
                    crossfit=raw_power[~chosen].mean((0,1)).clamp_min(1e-4)
                    oracle[chosen]=crossfit
                norm_density=oracle/oracle.mean(-1,keepdim=True)
                noise=torch.fft.fft(real_noise*rotation.conj()/rms,norm='ortho')
                sig=spectrum-noise
                frequency=256*torch.fft.fftfreq(spectrum.shape[-1])
                all_bands={}
                for bounds in BANDS:
                    seen=frequency.abs()>bounds[-1]
                    region=(frequency.abs()>=bounds[0])&(frequency.abs()<=bounds[1])
                    values={}
                    for readout,obs in (('received_witness',spectrum),('isolated_noise_witness',noise)):
                        guess=conditional_noise(obs,rotation,norm_density,seen)
                        values[readout]=measurements(.5*guess,noise,sig,rms,region,int(seen.sum()))
                    all_bands[band(bounds)]=values
                power_variation=(measured_profile-1).square().mean().sqrt()
                lags=torch.fft.ifft(measured_profile).abs()[1:9]
                all_views[scene]=dict(bands=all_bands,raw_psd_rms_shape=float(power_variation),
                                      first_eight_abs_acf=lags.tolist())
            report[route]=all_views
    print(json.dumps(dict(n=len(selected),role='held_U_s',fit_count_per_fold=120,
        diagnostic_only=True,F_reads=0,TX_reads=0,V_reads=0,target_reads=0,
        noise_used_in_any_deployable_candidate=False,registered=report)),flush=True)


if __name__=='__main__':
    main()
