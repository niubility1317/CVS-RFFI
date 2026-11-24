"""Summarize carrier trajectories on isolated, held source U packets."""
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.carrier_factorial_source import CAPTURE, factorial_readouts
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import FS_HZ, SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.physical_inverse_source import source_trace_implementation
from experiments.pamf_lora_main.runner import activate


def quantiles(value):
    flat = value.flatten()
    return dict(median=float(flat.median()), p90=float(flat.quantile(.9)))


def anatomy(tracker, stored):
    received, base, exact = tensor(stored).unbind(0)
    rotations = tuple(tracker.track(x)[1] for x in (base, exact, received))
    start, end, raw = rotations
    change = end * start.conj()
    midpoint = change[...,change.shape[-1]//2]
    center = midpoint.angle()
    displacement = (change[...,1:]*change[...,:-1].conj()).mean(-1).angle() * (FS_HZ/(2*np.pi))
    packet_mean = midpoint.mean(-1,keepdim=True).angle()
    private_center = torch.polar(torch.ones_like(center),center-packet_mean).angle()
    noisy_alignment = start * raw.conj()
    raw_center = noisy_alignment[...,noisy_alignment.shape[-1]//2].angle()
    raw_slope = (noisy_alignment[...,1:]*noisy_alignment[...,:-1].conj()).mean(-1).angle() * (FS_HZ/(2*np.pi))
    c = complex_iq(base)
    x = complex_iq(exact)
    aa = (c * start.conj()).mean(-1)
    ab = (x * end.conj()).mean(-1)
    gain_db = 20*(ab.abs().clamp_min(1e-7)/aa.abs().clamp_min(1e-7)).log10()
    polished = factorial_readouts(c,x,rotations)[-1]
    energy = (x-c).abs().square().sum((-1,-2))
    edit = (polished-c).abs().square().sum((-1,-2)) / energy.clamp_min(1e-12)
    return dict(abs_center_rad=quantiles(center.abs()),
                abs_center_after_common_rad=quantiles(private_center.abs()),
                abs_slope_hz=quantiles(displacement.abs()),
                abs_slope_after_packet_median_hz=quantiles((displacement-displacement.median(1,keepdim=True).values).abs()),
                abs_noisy_to_noiseless_center_rad=quantiles(raw_center.abs()),
                abs_noisy_to_noiseless_slope_hz=quantiles(raw_slope.abs()),
                abs_carrier_gain_db=quantiles(gain_db.abs()),
                projection_edit_energy_ratio=quantiles(edit))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    started = perf_counter()
    rt = read(Path('runtime.json'))
    activate(rt)
    source_trace_implementation()
    source = Path(rt['source_root'])
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source,contract,'U_s')
    training, cal, held = partition(rx)
    picked = np.concatenate([x[:30] for x in np.split(held,len(SOURCE))])
    if np.intersect1d(picked,np.concatenate((training,cal))).size:
        raise ValueError('Expected separate held U packets')
    jobs = [(raw[i],rx[i],ids[i]) for i in picked]
    tracker = PacketRestorer('consensus').eval()
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            results[route] = {}
            for scene in (VIEWS[3], VIEWS[-1]):
                captures, errors = zip(*pool.map(partial(CAPTURE,route,scene),jobs,chunksize=3),strict=True)
                results[route][scene] = anatomy(tracker,np.stack(captures,axis=1))
                print('LORA_CARRIER_ANATOMY',route,scene,len(captures),max(errors),flush=True)
    print(json.dumps(dict(status='COMPLETE',n=len(picked),seconds=perf_counter()-started,
                          target_reads=0,TX_reads=0,V_reads=0,F_reads=0,rows=results),allow_nan=False))


if __name__=='__main__':
    main()
