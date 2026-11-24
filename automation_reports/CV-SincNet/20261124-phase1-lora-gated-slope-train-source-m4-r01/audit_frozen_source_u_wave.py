from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.gated_slope_source import MODES,load_arms,render
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_band_risk_source import fft,private_projection
from experiments.pamf_lora_main.private_polar_source import wave_quantities
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate


@torch.no_grad()
def main():
    rt=read(Path('runtime.json'))
    activate(rt)
    assert rt['runner']=='experiments.pamf_lora_main.gated_slope_source'
    src=Path(rt['source_root'])
    root=Path(rt['run_root'])
    contract=source_contract(src)
    raw,rx,ids,_=source_role(src,contract,'U_s')
    _,_,held=partition(rx)
    indices=np.concatenate([p[:30] for p in np.split(held,len(SOURCE))])
    assert len(indices)==150
    models=load_arms(root,contract,'cpu')
    tracker=PacketRestorer('consensus').eval()
    modes=('direct','matched_half','pooled_half','inference_only_half','matched_full','pooled_full')
    offsets=[MODES.index(mode) for mode in modes]
    result={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            result[route]={}
            for scene in (VIEWS[3],VIEWS[-1]):
                jobs=((route,scene,raw[idx],rx[idx],ids[idx]) for idx in indices)
                iq,epsilon=(tensor(np.stack(v)) for v in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                observed=[]
                active=0
                for chunk in iq.split(18):
                    output,n=render(chunk,None,lambda y:(y,None),tracker,models)
                    observed.append(output[offsets])
                    active+=n
                observations=torch.cat(observed,dim=1)
                # Same fixed, noise-free received-coordinate reference for all output rows.
                y=iq-epsilon
                expected,rotation,_,rms=coordinates(tracker,y)
                reference=private_projection(fft(expected))
                bands=named_band_masks(expected).float()
                result[route][scene]=dict(n=len(iq),active=active,rows={
                    name:wave_quantities(private_projection(fft(complex_iq(obs)*rotation.conj()/rms)),
                        reference,rms.square(),bands) for name,obs in zip(modes,observations,strict=True)})
                print(f'LO_RA_FROZEN_SOURCE_WAVE {route} {scene} {active}',flush=True)
    print(json.dumps(dict(n=150,status='PASS',F_reads=0,V_reads=0,TX_reads=0,target_reads=0,
          reference_used_after_render=True,rows=result)),flush=True)


if __name__=='__main__':
    main()
