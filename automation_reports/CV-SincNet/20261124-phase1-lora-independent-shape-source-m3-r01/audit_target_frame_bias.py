"""Measure deterministic carrier leakage in the previous complex-shape target."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read,tensor
from experiments.pamf_lora_main.carrier_coupled import observed_frame
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.full_packet_views import source_contract,source_role
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.private_transport_bands import named_band_masks
from experiments.pamf_lora_main.runner import activate


def private_spectrum(value):
    private = value-value.mean(1,keepdim=True)
    return torch.fft.fft(private,norm='ortho')


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    root = Path(rt['source_root'])
    c = source_contract(root)
    raw,rx,ids,_ = source_role(root,c,'U_s')
    fit,cal,held = partition(rx)
    idx = np.concatenate([a[:30] for a in np.split(held,len(SOURCE))])
    if np.intersect1d(idx,np.concatenate((fit,cal))).size:
        raise ValueError('Expected held source U')
    tracker=PacketRestorer('consensus').eval()
    clean = tensor(np.stack([chirp_windows(raw[i]) for i in idx]))
    original,rotation,_,_ = observed_frame(tracker,clean)
    clean_spectrum = private_spectrum(original)
    carrier = original.mean(-1,keepdim=True)
    masks = named_band_masks(original)
    rows = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            rows[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs=((route,view,raw[i],rx[i],ids[i]) for i in idx)
                capture,_=(tensor(np.stack(a)) for a in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                _,received_rotation,_,_=observed_frame(tracker,capture)
                delta=rotation*received_rotation.conj()
                leakage=private_spectrum(carrier*delta)
                wrong=private_spectrum(original*delta)-clean_spectrum
                ratios=[]
                for mask in masks:
                    power=clean_spectrum[...,mask].abs().square().sum((1,2)).clamp_min(1e-9)
                    relative=leakage[...,mask].abs().square().sum((1,2))/power
                    mismatch=wrong[...,mask].abs().square().sum((1,2))/power
                    ratios.append(dict(leakage_median=float(relative.median()),mismatch_median=float(mismatch.median()),
                                       fraction_above_detail=float((relative>1).float().mean())))
                rows[route][view]=ratios
                print('LORA_SHAPE_TARGET_BIAS',route,view,len(idx),flush=True)
    print(json.dumps(dict(status='COMPLETE',n=len(idx),target_reads=0,F_reads=0,V_reads=0,TX_reads=0,rows=rows),allow_nan=False))


if __name__=='__main__':
    main()
