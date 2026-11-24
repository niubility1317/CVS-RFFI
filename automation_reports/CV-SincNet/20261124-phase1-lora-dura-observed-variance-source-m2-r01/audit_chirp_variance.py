"""Validate the fixed off-band estimate per chirp across held source views."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.carrier_coupled import packet_frame
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_observed_variance_source import RUN, receive_variance
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.gated_slope import THRESHOLD
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.runner import activate


def correlation(a,b):
    a,b = a-a.mean(),b-b.mean()
    norm = a.norm()*b.norm()
    return float((a*b).sum()/norm) if norm>0 else None


def measurements(estimate, actual):
    estimate_log, actual_log = estimate.log(), actual.log()
    per_packet = correlation(estimate.mean(1).log(),actual.mean(1).log())
    x = (estimate_log-estimate_log.mean(1,keepdim=True)).flatten()
    y = (actual_log-actual_log.mean(1,keepdim=True)).flatten()
    quantiles = torch.quantile((estimate/actual).flatten(),torch.tensor([.1,.5,.9])).tolist()
    return dict(n_packets=len(estimate),packet_log_correlation=per_packet,
                within_packet_log_correlation=correlation(x,y), per_chirp_ratios=quantiles)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = (Path(rt[k]) for k in ('run_root','source_root'))
    if root.name != RUN or read(root/'smoke.json')['status'] != 'PASS':
        raise ValueError('Frozen receive estimator and source smoke required')
    contract=source_contract(source)
    iq, rx, ids, _=source_role(source,contract,'U_s')
    a,b,held=partition(rx)
    chosen=np.concatenate([r[:30] for r in np.split(held,len(SOURCE))])
    if np.intersect1d(chosen,np.concatenate((a,b))).size:
        raise ValueError('Calibration diagnostic uses held source U only')
    tracker=PacketRestorer('consensus').eval()
    clean=tensor(np.stack([chirp_windows(iq[i]) for i in chosen]))
    y,_,_,_=packet_frame(coordinates(tracker,clean))
    clean_power=2*receive_variance(y)
    results={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            results[route]={}
            for view in VIEWS[1:]:
                jobs=((route,view,iq[i],rx[i],ids[i]) for i in chosen)
                observed, noise=(tensor(np.stack(x)) for x in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                frame=coordinates(tracker,observed)
                active=corruption_score(frame[0])>THRESHOLD
                y,_,_,rms=packet_frame(frame)
                expected=2*receive_variance(y).reshape(y.shape[:2])
                actual=(complex_iq(noise)/rms).abs().square().mean(-1)
                results[route][view]=dict(all=measurements(expected,actual),active=int(active.sum()),
                      active_measurements=measurements(expected[active],actual[active]) if active.sum()>=3 else None)
                print('LORA_DURA_RECEIVE_VARIANCE_U',route,view,int(active.sum()),flush=True)
    print(json.dumps(dict(status='VERIFIED',results=results,clean_complex_variance_quantiles=torch.quantile(
        clean_power,torch.tensor([.5,.95])).tolist(),n=len(chosen),target_reads=0,F_reads=0,V_reads=0,TX_reads=0,
        exact_noise_diagnostic_only=True),allow_nan=False))


if __name__ == '__main__':
    main()
