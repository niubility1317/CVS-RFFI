"""Isolate the runtime effect of cross-chirp attention at fixed trained weights."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.antithetic_detail_source import bind_models
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_spectral_context_source import RECIPES
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rt=read(Path('runtime.json'))
    activate(rt)
    root,source=Path(rt['run_root']),Path(rt['source_root'])
    contract=source_contract(source)
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held=partition(rx)
    picked=np.concatenate([part[:30] for part in np.split(held,len(SOURCE))])
    tracker=PacketRestorer('consensus').eval()
    _,arms,threshold=bind_models(root,contract,'cpu',recipes=RECIPES)
    model=arms[1]
    data={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            jobs=((route,VIEWS[-1],raw[i],rx[i],ids[i]) for i in picked)
            batch=tensor(np.stack([iq for iq,_ in pool.map(replay,jobs,chunksize=6)]))
            output=[]
            for part in batch.split(12):
                y,rot,q,rms=coordinates(tracker,part)
                active=corruption_score(y)>threshold
                if not active.any():
                    continue
                y,rot,q,rms=(a[active] for a in (y,rot,q,rms))
                model.spectral_mixer.packet=True
                restored=model(y,q,rot)[-1]
                model.spectral_mixer.packet=False
                within=model(y,q,rot)[-1]
                axes=named_band_masks(y)[:2]
                delta=torch.fft.fft(restored-within,norm='ortho')*rms
                original=torch.fft.fft(restored-y,norm='ortho')*rms
                bands=(axes[0],axes[1],torch.ones(y.shape[-1],dtype=torch.bool))
                output.append(torch.stack([torch.linalg.vector_norm(delta[...,m],dim=(-2,-1)) /
                                           torch.linalg.vector_norm(original[...,m],dim=(-2,-1)).clamp_min(1e-9)
                                           for m in bands],dim=-1))
            values=torch.cat(output)
            data[route]=dict(active=len(values),edit_ratio_p10_p50_p90=
                              torch.quantile(values,values.new_tensor([.1,.5,.9]),dim=0).tolist())
    print(json.dumps(dict(status='PASS',n=150,scope=['1_to_8','9_to_32','all'],target_reads=0,F_reads=0,rows=data)))


if __name__=='__main__':
    main()
