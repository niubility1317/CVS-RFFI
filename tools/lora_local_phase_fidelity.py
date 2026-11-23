"""Audit frozen LoRa recovery against paired source waveforms and traces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main import local_phase_source, spectral_block_source
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.local_wiener_source import ROUTES, read_frozen
from experiments.pamf_lora_main.runner import activate


BANDS = ((1, 8), (9, 32), (33, 64), (65, 128))
METRICS = ('complex_nmse', 'magnitude_nmse', 'normalized_real_correlation')
GEOMETRY_RUN = '20261124-phase1-lora-complex-geometry-source-m4-r01'
TANGENT_RUN = '20261124-phase1-lora-empirical-tangent-source-m4-r01'


def bounded_projection(model, direct, reference, y, frame, threshold):
    """L2 projection onto the candidate's exact noncarrier editing disks."""
    x, _, _, level, _ = model.observe(y)
    target = model.spectral(reference/frame)
    bound = .60*(x.abs()+.15*level.clamp_min(1e-8).sqrt())
    correction = model.bounded(target-x,bound)
    output = direct+model.temporal(correction)*frame
    use = corruption_score(y)>threshold
    return torch.where(use[:,None,None],output,direct)


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


def direction_fraction(model, tracked, frame, waves):
    direct, *predictions = waves
    edits = torch.fft.fft((torch.stack(predictions)-direct)/frame,norm='ortho')
    actual = torch.fft.fft(tracked,norm='ortho')
    power = actual[...,model.outer].abs().square().mean(-1,keepdim=True)
    numerator, denominator = [], []
    for name, support, destination in (('even',slice(0,None,2),slice(1,None,2)),
                                        ('odd',slice(1,None,2),slice(0,None,2))):
        _, _, atoms, dual = model.query(actual[:,support],power[:,support],name,destination)
        move = edits[:,:,destination][...,model.weak].flatten(2).unsqueeze(-1)
        projection = atoms@(dual@move)
        numerator.append(projection.abs().square().sum((-1,-2)))
        denominator.append(move.abs().square().sum((-1,-2)))
    ratio = (numerator[0]+numerator[1])/(denominator[0]+denominator[1]).clamp_min(1e-10)
    dose = edits.abs().square().mean((-1,-2)).sqrt()
    return np.asarray(torch.stack((ratio,dose),dim=-1).tolist(),dtype=np.float32)


def paired_source(data, cache, contract, root, route, modes):
    raw, receivers, ids, _ = source_role(data,contract,'V')
    selected = panel(receivers)
    frozen_ids, _, marks = read_frozen(root,route,modes)
    if ids[selected].tolist()!=frozen_ids:
        raise ValueError('The diagnostic must follow the same frozen source V physical IDs')
    rows = list(zip(raw[selected],receivers[selected],ids[selected],strict=True))
    registered = np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    clean = registered[0,selected]
    return rows, registered, clean, selected, marks


@torch.no_grad()
def diagnose(rt, route, device):
    is_geometry = rt['run_id']==GEOMETRY_RUN
    is_tangent = rt['run_id']==TANGENT_RUN
    if is_geometry or is_tangent:
        if is_tangent:
            from experiments.pamf_lora_main import empirical_tangent_source as study
            recipes = study.recipes(rt)
        else:
            from experiments.pamf_lora_main import complex_geometry_source as study
            recipes = study.RECIPES
        recover = spectral_block_source.recover
        readouts = ('direct',*study.MODES[2:])
        indices = range(len(study.MODES)-2)
    else:
        study = local_phase_source
        recipes = study.RECIPES
        recover = study.recover
        readouts = ('direct','middle_full','edges_full','edges_half')
        indices = (1,4,5)
    root = Path(rt['run_root'])
    data = Path(rt['source_root'])
    cache = root.parent/DATA_RUN/'source_views'
    contract = source_contract(data)
    rows, registered, clean, selected, marks = paired_source(data,cache,contract,root,route,study.MODES)
    models, threshold = spectral_block_source.load_models(root,contract,device,recipes)
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
            direction = []
            for received, epsilon, source in zip(tensor(capture,device).split(18),tensor(noise,device).split(18),truth.split(18),strict=True):
                recoveries, count = recover(received,tracker,models,threshold)[:2]
                active += count
                y = complex_iq(received)
                waves = [y,*(complex_iq(recoveries[k]) for k in indices)]
                tracked, phase, _, rms = coordinates(tracker,received)
                frame = phase*rms
                if is_tangent:
                    direction.append(direction_fraction(models[-1],tracked,frame,waves))
                # One fit against the direct readout is shared by all diagnostic comparisons.
                ground = complex_iq(source)
                gain = (y*ground.conj()).mean(-1,keepdim=True)/ground.abs().square().mean(-1,keepdim=True).clamp_min(1e-6)
                target = ground*gain
                noiseless = y-complex_iq(epsilon)
                for reference, group in ((target,bands_from_ground),(noiseless,bands_from_noiseless)):
                    compared = waves
                    if is_geometry:
                        projected = bounded_projection(models[-1],y,reference,tracked,frame,threshold)
                        compared = [*waves,projected]
                    group.append(spectral_metrics(torch.stack(compared),reference,frame))
            if active!=marks[view]['active']:
                raise ValueError('Diagnostic gate triggers differ from frozen predictions')
            ground, noiseless = [np.concatenate(parts,axis=1) for parts in (bands_from_ground,bands_from_noiseless)]
            output[view] = {'active':active}
            for key, values in (('aligned_source',ground),('noiseless_reception',noiseless)):
                output[view][key] = dict(median=np.median(values,axis=1).tolist(),
                    by_receiver={name:np.median(group,axis=1).tolist()
                                 for name,group in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
            if is_tangent:
                values = np.concatenate(direction,axis=1)
                output[view]['direction'] = dict(readouts=readouts[1:],fields=['projected_weak_fraction','rms_dose'],
                    median=np.median(values,axis=1).tolist(),
                    by_receiver={name:np.median(group,axis=1).tolist()
                                 for name,group in zip(SOURCE,np.split(values,len(SOURCE),axis=1),strict=True)})
    if is_geometry:
        readouts = (*readouts,'bounded_pair_projection')
    return dict(route=route,n=len(rows),bands=BANDS,metrics=METRICS,readouts=readouts,scenes=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('route',choices=ROUTES)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    rt = read(args.runtime)
    root = Path(rt['run_root'])
    if rt.get('run_id') not in (local_phase_source.RUN,GEOMETRY_RUN,TANGENT_RUN) or read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('The full dual-route truth-last scoring must already be complete')
    activate(rt)
    print(json.dumps(diagnose(rt,args.route,args.device),allow_nan=False))


if __name__=='__main__':
    main()
