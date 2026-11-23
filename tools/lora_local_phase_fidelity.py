"""Audit frozen LoRa phase readouts against paired source waveforms and traces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.local_phase_source import MODES, RECIPES, RUN, recover
from experiments.pamf_lora_main.local_wiener_source import ROUTES, read_frozen
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.spectral_block_source import load_models


BANDS = ((1, 8), (9, 32), (33, 64), (65, 128))
METRICS = ('complex_nmse', 'magnitude_nmse', 'normalized_real_correlation')
READOUTS = ('direct', 'middle_full', 'edges_full', 'edges_half')


def spectral_metrics(waves, known, frame):
    spectrum = torch.fft.fft(waves/frame,norm='ortho')
    truth = torch.fft.fft(known/frame,norm='ortho')
    bins = torch.arange(256,device=waves.device)
    bins = torch.minimum(bins,256-bins)
    bands = []
    for low,high in BANDS:
        use = (bins>=low)&(bins<=high)
        received, reference = spectrum[...,use],truth[...,use]
        power = reference.abs().square().sum((-2,-1)).clamp_min(1e-6)
        error = (received-reference).abs().square().sum((-2,-1))/power
        magnitude = (received.abs()-reference.abs()).square().sum((-2,-1))/power
        product = (received*reference.conj()).sum((-2,-1)).real
        coherence = product/(power*received.abs().square().sum((-2,-1)).clamp_min(1e-6)).sqrt()
        bands.append(torch.stack((error,magnitude,coherence),dim=-1))
    return np.asarray(torch.stack(bands,dim=2).tolist(),dtype=np.float32)


def paired_source(data, cache, contract, root, route):
    raw, receivers, ids, _ = source_role(data,contract,'V')
    selected = panel(receivers)
    frozen_ids, _, marks = read_frozen(root,route,MODES)
    if ids[selected].tolist()!=frozen_ids:
        raise ValueError('The diagnostic must follow the same frozen source V physical IDs')
    rows = list(zip(raw[selected],receivers[selected],ids[selected],strict=True))
    registered = np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    clean = registered[0,selected]
    return rows, registered, clean, selected, marks


@torch.no_grad()
def diagnose(rt, route, device):
    root = Path(rt['run_root'])
    data = Path(rt['source_root'])
    cache = root.parent/DATA_RUN/'source_views'
    contract = source_contract(data)
    rows, registered, clean, selected, marks = paired_source(data,cache,contract,root,route)
    models, threshold = load_models(root,contract,device,RECIPES)
    tracker = PacketRestorer('consensus').to(device).eval()
    truth = tensor(clean,device)
    output = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, view in enumerate(VIEWS):
            capture, noise = (np.stack(v) for v in zip(*pool.map(partial(replay,route,view),rows,chunksize=8),strict=True))
            if route=='full' and not np.array_equal(capture,registered[i,selected]):
                raise ValueError('Full source trace replay changed registered IQ')
            if route=='residual':
                for j in range(0,len(rows),len(rows)//len(SOURCE)):
                    if not np.array_equal(capture[j],receive_only(route,view,rows[j])[0]):
                        raise ValueError('Residual source trace replay changed sampled registered IQ')
            active, bands_from_ground, bands_from_noiseless = 0, [], []
            for received, epsilon, source in zip(tensor(capture,device).split(18),tensor(noise,device).split(18),truth.split(18),strict=True):
                recoveries, count, _ = recover(received,tracker,models,threshold)
                active += count
                y = complex_iq(received)
                waves = torch.stack([y,*(complex_iq(recoveries[k]) for k in (1,4,5))])
                _, phase, _, rms = coordinates(tracker,received)
                frame = phase*rms
                # One fit against the direct readout is shared by all diagnostic comparisons.
                ground = complex_iq(source)
                gain = (y*ground.conj()).mean(-1,keepdim=True)/ground.abs().square().mean(-1,keepdim=True).clamp_min(1e-6)
                bands_from_ground.append(spectral_metrics(waves,ground*gain,frame))
                bands_from_noiseless.append(spectral_metrics(waves,y-complex_iq(epsilon),frame))
            if active!=marks[view]['active']:
                raise ValueError('Diagnostic gate triggers differ from frozen predictions')
            ground, noiseless = [np.concatenate(parts,axis=1) for parts in (bands_from_ground,bands_from_noiseless)]
            output[view] = {'active':active}
            for key, values in (('aligned_source',ground),('noiseless_reception',noiseless)):
                output[view][key] = dict(median=np.median(values,axis=1).tolist(),
                    by_receiver={name:np.median(group,axis=1).tolist()
                                 for name,group in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
    return dict(route=route,n=len(rows),bands=BANDS,metrics=METRICS,readouts=READOUTS,scenes=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('route',choices=ROUTES)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    rt = read(args.runtime)
    root = Path(rt['run_root'])
    if rt.get('run_id')!=RUN or read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('The full dual-route truth-last scoring must already be complete')
    activate(rt)
    print(json.dumps(diagnose(rt,args.route,args.device),allow_nan=False))


if __name__=='__main__':
    main()
