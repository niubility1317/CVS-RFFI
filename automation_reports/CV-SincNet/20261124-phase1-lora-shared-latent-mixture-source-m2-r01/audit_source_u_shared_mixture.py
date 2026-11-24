from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer,complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract,source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_band_risk_source import fft
from experiments.pamf_lora_main.private_polar_source import wave_quantities
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.shared_latent_mixture_source import MODES,bound_assets,readouts

@torch.no_grad()
def main():
    rt=read(Path('runtime.json'))
    activate(rt)
    assert rt['runner']=='experiments.pamf_lora_main.shared_latent_mixture_source'
    root=Path(rt['run_root'])
    source=Path(rt['source_root'])
    contract=source_contract(source)
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held=partition(rx)
    selected=np.concatenate([group[:30] for group in np.split(held,len(SOURCE))])
    (model,centers),gate=bound_assets(root,contract,'cpu')
    tracker=PacketRestorer('consensus').eval()
    result={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            views=result[route]={}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs=((route,view,raw[i],rx[i],ids[i]) for i in selected)
                observations,epsilon=(tensor(np.stack(x)) for x in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                spectra={mode:[] for mode in ('direct',*MODES[2:])}
                bases,weights=[],[]
                active=0
                for wave in observations.split(18):
                    transformed,n=readouts(wave,tracker,model,gate,centers=centers)
                    _,rot,_,rms=coordinates(tracker,wave)
                    current=rot*rms
                    bases.append(current)
                    weights.append(rms.square().mean(1,keepdim=True))
                    for name,value in zip(spectra,(wave,*transformed),strict=True):
                        spectra[name].append(fft(complex_iq(value)/current).mean(1,keepdim=True))
                    active+=n
                # Only now form the isolated comparison with the new noise removed.
                truth=fft(complex_iq(observations-epsilon)/torch.cat(bases)).mean(1,keepdim=True)
                masks=named_band_masks(truth).float()
                weight=torch.cat(weights)
                views[view]=dict(n=len(selected),active=active,
                    by_mode={name:wave_quantities(torch.cat(rows),truth,weight,masks) for name,rows in spectra.items()})
                print('LORA_SHARED_MIXTURE_U_WAVE',route,view,active,flush=True)
    print(json.dumps(dict(status='PASS',n=len(selected),isolated_reference_after_prediction=True,
       F_reads=0,V_reads=0,TX_reads=0,target_reads=0,rows=result)),flush=True)

if __name__=='__main__':
    main()
