"""Measure staged edit and additive-noise directions on held LoRa source U only."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from .antithetic_detail_source import load_arm
from .carrier_coupled import packet_frame
from .channel_blind import corruption_score
from .channel_blind_source import partition
from .data import SOURCE, VIEWS
from .dura_baseband_source import CONTROL, REUSED, RUN
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates
from .offband_noise_source import replay
from .relative_carrier import match_packet_power, mix
from .runner import activate


def direction_parts(edit, noise):
    axes = (-2, -1)
    energy = noise.abs().square().sum(axes, keepdim=True).clamp_min(1e-10)
    removed = -(edit * noise.conj()).real.sum(axes, keepdim=True) / energy
    orthogonal = (edit + removed * noise).abs().square().sum(axes, keepdim=True) / energy
    remaining = (edit + noise).abs().square().sum(axes, keepdim=True) / energy
    return removed.flatten(), orthogonal.flatten(), remaining.flatten()


def private_near(value):
    frequency = torch.fft.fftfreq(value.shape[-1], device=value.device).abs()
    band = (frequency > 0) & (frequency <= 8 / value.shape[-1])
    return torch.fft.fft(value - value.mean(1,keepdim=True), norm='ortho')[...,band]


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = (Path(rt[key]) for key in ('run_root', 'source_root'))
    if root.name != RUN:
        raise ValueError('Expected the registered dechirped DURA run')
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    fit, cal, held = partition(rx)
    chosen = np.concatenate([part[:30] for part in np.split(held,len(SOURCE))])
    if np.intersect1d(chosen,np.concatenate((fit,cal))).size:
        raise ValueError('Only held source packets are permitted')
    control, threshold = load_arm(root, CONTROL, contract, 'cpu', REUSED)
    tracker = PacketRestorer('consensus').eval()
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            results[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs = ((route,view,raw[i],rx[i],ids[i]) for i in chosen)
                iq, exact_noise = (tensor(np.stack(x)) for x in
                                   zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                frame = coordinates(tracker,iq)
                active = corruption_score(frame[0]) > threshold
                y, rotation, _, scale = packet_frame(tuple(v[active] for v in frame))
                native = real_iq(y * rotation).flatten(0,1)
                stages, _, _, coarse = control.solve(native,y,rotation)
                proposals = [mix(y,match_packet_power(
                    complex_iq(value.unflatten(0,y.shape[:2])) * rotation.conj(),y),.25)
                    for value in (coarse, *stages)]
                # The actual noise and isolated no-added-noise reference enter only here.
                noise = complex_iq(exact_noise[active]) * rotation.conj() / scale
                true_near = private_near(y - noise)
                raw_near_error = (private_near(y) - true_near).abs().square().sum((1,2)).clamp_min(1e-10)

                def summarize(piece):
                    return torch.quantile(piece,piece.new_tensor([.1,.5,.9])).tolist()

                rows = {}
                for name, wave in zip(('coarse','first_map','second_map'),proposals,strict=True):
                    removed, orthogonal, remaining = direction_parts(wave-y,noise)
                    relative = (private_near(wave) - true_near).abs().square().sum((1,2)) / raw_near_error
                    rows[name] = dict(removed_noise_fraction=summarize(removed),
                                      orthogonal_edit_over_noise=summarize(orthogonal),
                                      total_error_over_raw=summarize(remaining),
                                      private_near_error_over_raw=summarize(relative))
                results[route][view] = dict(active=int(active.sum()),stages=rows)
                print('LORA_DURA_STAGE_NOISE_U',route,view,int(active.sum()),flush=True)
    print(json.dumps(dict(status='VERIFIED',physical_count=len(chosen),dose=.25,
                          trained_control=REUSED[CONTROL.name],results=results,
                          F_reads=0,V_reads=0,TX_reads=0,target_reads=0,
                          true_noise_usage='isolated_direction_diagnostic_only'),allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
