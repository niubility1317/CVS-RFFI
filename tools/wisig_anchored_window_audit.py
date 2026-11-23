"""Source-only timing and waveform attribution after WiSig truth-last scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_unrolled_restore import wisig_anchored_window_source as study
from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex


def timing_reference(received, clean, positions, offsets):
    valid = (positions>=0)&(positions<received.shape[-1])
    observed = received[:,positions.clamp(0,received.shape[-1]-1)]
    target = clean[:,None]*valid
    observed = observed*valid
    product = (target.conj()*observed).sum(-1).abs().square()
    scale = target.abs().square().sum(-1)*observed.abs().square().sum(-1)
    support = (product / scale.clamp_min(1e-8)).clamp_max(1.)
    best,index = support.max(dim=1)
    return offsets[index],best


def waveform_metrics(candidate, reference, seen, frequency):
    target = torch.fft.fft(reference,norm='ortho')
    spectrum = torch.fft.fft(candidate,norm='ortho')
    bands = (frequency<=.25,(frequency>.25)&(frequency<=.40),frequency>.40)
    values = []
    for mask in bands:
        x,y = spectrum[:,mask],target[:,mask]
        signal = y.abs().square().sum(-1).clamp_min(1e-8)
        values.extend(((x-y).abs().square().sum(-1)/signal,
                       (x.abs()-y.abs()).square().sum(-1)/signal,
                       (x*y.conj()).sum(-1).real / (signal*x.abs().square().sum(-1).clamp_min(1e-8)).sqrt()))
    for mask in (seen,~seen):
        power = (reference.abs().square()*mask).sum(-1).clamp_min(1e-8)
        values.append(((candidate-reference).abs().square()*mask).sum(-1)/power)
    return torch.stack(values,dim=-1)


FIELDS = tuple(f'{band}_{name}' for band in ('0-6.25MHz','6.25-10MHz','10-12.5MHz')
               for name in ('complex_nmse','amplitude_nmse','real_coherence')) + ('overlap_nmse','edge_nmse')
TIMING_FIELDS = ('reference_shift','reference_squared_coherence','estimated_shift','absolute_shift_error')


@torch.no_grad()
def diagnose(rt, route):
    root = Path(rt['run_root'])
    if rt['run_id']!=study.RUN or read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Only the completed independent WiSig source experiment may be audited')
    cache,_,_,_,ids,rx,_ = study.core.baseline.handles(rt)
    device = 'cuda:0'
    models = tuple(study.load_models(rt))
    receiver = {str(k):tensor(np.flatnonzero(rx==k),device).long() for k in np.unique(rx)}
    receiver['all'] = torch.arange(len(ids),device=device)
    time = torch.arange(256,device=device)
    offsets = torch.arange(-64,65,device=device)
    positions = time[None]+offsets[:,None]
    frequency = torch.fft.fftfreq(256,device=device).abs()
    directory = cache/study.ROUTES[route]
    clean = as_complex(tensor(np.load(directory/'clean.npy',mmap_mode='r',allow_pickle=False),device))
    result = {}
    for view in study.core.baseline.VIEWS:
        incoming = as_complex(tensor(np.load(directory/f'{view}.npy',mmap_mode='r',allow_pickle=False),device))
        output = {arm:[] for arm in study.ARMS}
        shifts = {arm:[] for arm in study.ARMS}
        for received, ref in zip(incoming.split(48),clean.split(48),strict=True):
            expected,coherence = timing_reference(received,ref,positions,offsets)
            for arm,(model,_) in zip(study.ARMS,models,strict=True):
                stages,estimated,_,_ = model(received)
                moves = time+estimated[:,None]
                seen = (moves>=0)&(moves<256)
                output[arm].append(torch.stack([waveform_metrics(z,ref,seen,frequency) for z in (received,*stages)]))
                shifts[arm].append(torch.stack((expected,coherence,estimated,(estimated-expected).abs()),dim=-1))
        result[view] = {}
        for arm in study.ARMS:
            waveform = torch.cat(output[arm],dim=1)
            timing = torch.cat(shifts[arm])
            result[view][arm] = {}
            for group,selection in receiver.items():
                t = timing.index_select(0,selection)
                w = waveform.index_select(1,selection)
                edge = w[...,t[:,2]!=0,-1]
                # Missing-edge metrics require at least one extrapolated sample.
                edge_median = edge.median(dim=1).values.tolist() if edge.shape[1] else None
                result[view][arm][group] = dict(n=len(selection),timing_median=t.median(dim=0).values.tolist(),
                    error_at_most_two=int((t[:,3]<=2).sum()),reference_outside_search=int((t[:,0].abs()>40).sum()),
                    waveform_median=w[...,:-1].median(dim=1).values.tolist(),edge_median=edge_median)
    return dict(route=route,n=len(ids),target_reads=0,TX_truth_reads=0,F_reads=0,
                timing_fields=TIMING_FIELDS,waveform_fields=FIELDS,readouts=('direct','timed','coarse','joint'),views=result)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('route',choices=study.ROUTES)
    args = parser.parse_args()
    rt = read(args.runtime)
    activate(rt)
    print(json.dumps(diagnose(rt,args.route),allow_nan=False))
