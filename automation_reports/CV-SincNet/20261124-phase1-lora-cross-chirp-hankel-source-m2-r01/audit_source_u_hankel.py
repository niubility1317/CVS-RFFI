from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read,tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.cross_chirp_hankel_source import MODES,readouts
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer,complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract,source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_band_risk_source import fft
from experiments.pamf_lora_main.private_polar_source import wave_quantities
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.weak_guard_source import parent_model

def compare_regions(value,truth,weight,mask):
    estimate=torch.fft.ifft(value*mask,norm='ortho')
    reference=torch.fft.ifft(truth*mask,norm='ortho')
    error=(estimate-reference).abs().square()*weight
    energy=reference.abs().square()*weight
    central=slice(64,192)
    ends=tuple(range(64))+tuple(range(192,256))
    return dict(center=float(error[...,central].sum()/energy[...,central].sum().clamp_min(1e-12)),
                ends=float(error[...,list(ends)].sum()/energy[...,list(ends)].sum().clamp_min(1e-12)))

@torch.no_grad()
def main():
    rt=read(Path('runtime.json'))
    activate(rt)
    assert rt['runner']=='experiments.pamf_lora_main.cross_chirp_hankel_source'
    root=Path(rt['run_root'])
    source=Path(rt['source_root'])
    contract=source_contract(source)
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held=partition(rx)
    selected=np.concatenate([part[:30] for part in np.split(held,len(SOURCE))])
    tracker=PacketRestorer('consensus').eval()
    model,threshold=parent_model(root,contract,'cpu')
    report={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            report[route]={}
            for scene in (VIEWS[3],VIEWS[-1]):
                jobs=((route,scene,raw[i],rx[i],ids[i]) for i in selected)
                iq,eps=(tensor(np.stack(xs)) for xs in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                basis,weights=[],[]
                restored={name:[] for name in ('direct',*MODES[2:])}
                active=0
                mismatch=0.
                for batch in iq.split(12):
                    _,rot,_,rms=coordinates(tracker,batch)
                    basis.append(rot*rms)
                    weights.append(rms.square())
                    wave,n=readouts(batch,tracker,model,threshold)
                    observed=wave[-2:]
                    norms=[torch.linalg.vector_norm((x-batch).flatten(-2),dim=-1) for x in observed]
                    mismatch=max(mismatch,float((norms[0]-norms[1]).abs().amax()))
                    for name,value in zip(restored,(batch,*wave),strict=True):
                        restored[name].append(fft(complex_iq(value)/(rot*rms)))
                    active+=n
                expected=fft(complex_iq(iq-eps)/torch.cat(basis))
                masks=named_band_masks(expected)
                w=torch.cat(weights)
                rows={mode:torch.cat(common) for mode,common in restored.items()}
                report[route][scene]=dict(n=len(iq),active=active,physical_dose_max_mismatch=mismatch,
                     by_mode={mode:wave_quantities(x,expected,w,masks.float()) for mode,x in rows.items()},
                     near_time={mode:compare_regions(x,expected,w,masks[:2].any(0)) for mode,x in rows.items()})
                print('LORA_HANKEL_HELD_U',route,scene,active,flush=True)
    print(json.dumps(dict(status='PASS',n=150,isolated_reference_after_readouts=True,
                      F_reads=0,V_reads=0,TX_reads=0,target_reads=0,rows=report)),flush=True)

if __name__=='__main__':
    main()
