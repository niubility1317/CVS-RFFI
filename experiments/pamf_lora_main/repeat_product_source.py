"""Frozen engineering comparison for cross-chirp common-band directions."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from . import cross_frequency_source as parent
from .band_allocation import band_masks, ifft, tracked_anchor
from .channel_blind import corruption_score
from .channel_blind_source import setup
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates
from .local_wiener_source import main
from .phase_dose_bursts_source import burst_observation
from .repeat_product import independent_noise, moments, normalized_spectrum, real_parts, repeat_features, solve
from .variance_transport import physical_norm
from .weak_sideband import limited_turn


RUN = '20261124-phase1-lora-repeat-product-source-m2-r01'
DEVICE = 'cuda:0'
MODES = ('direct', 'reference', 'anchor', 'linear_common', 'products_common')
COMPARISONS = (('products_common', 'linear_common'),
               *((mode, compare) for mode in MODES[2:] for compare in ('direct','reference','anchor')
                 if mode != compare))
FIELDS = ('active','anchor_rms','product_rms','private_change_rms','iso_weak_error')
SOURCE_ONLY = dict(fit=20000, gate=2500, held=500, optimizer_updates=0, F_reads=0, V_reads=0, TX_reads=0,
                   target_reads=0, exact_reads=0, practical_reads=0)


def desired_bins(device):
    return torch.tensor([*range(-32,0), *range(1,33)], device=device).remainder(256)


def input_features(tracked, desired):
    spectrum = normalized_spectrum(tracked)
    linear, products = repeat_features(spectrum, desired)
    return real_parts(linear), real_parts(torch.cat((linear, products), -1))


@torch.no_grad()
def calibrate(rt):
    started = perf_counter()
    (fitting, calibration, hold), contract = setup(rt, DEVICE, fit_per_rx=4000)
    if (len(fitting),len(calibration),len(hold)) != (20000,2500,500):
        raise ValueError('Expected the registered source U roles')
    tracker = PacketRestorer('consensus').to(DEVICE).eval()
    root = Path(rt['run_root'])
    ancestor = root.parent/parent.RUN
    if read(ancestor/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the registered frozen analytic prior')
    threshold = parent.load_gate(ancestor)
    parent.load(ancestor, contract, threshold)
    rng = torch.Generator(device=DEVICE).manual_seed(2026112449)
    desired = desired_bins(DEVICE)
    joint = [0,0]
    count = 0
    for batch in fitting.split(96):
        clean = complex_iq(batch)
        source = normalized_spectrum(coordinates(tracker, batch)[0])
        labels = real_parts(source[..., desired].unflatten(1,(2,3)).permute(0,2,3,1))
        for perturb in (None, independent_noise, burst_observation):
            y = batch if perturb is None else real_iq(perturb(clean, rng))
            features = input_features(coordinates(tracker, y)[0], desired)
            for i, arm in enumerate(features):
                value, _ = moments(arm, labels)
                joint[i] = joint[i] + value
            count += len(batch)
    design = tuple(solve(matrix, count, width) for matrix,width in zip(joint,(16,32),strict=True))
    if not all(torch.isfinite(value).all() for group in design for value in group):
        raise ValueError('Nonfinite real-IQ source covariance')
    torch.save(dict(contract=contract, threshold=threshold, design=design),root/'calibration.pt')
    write(root/'calibration.json',dict(**SOURCE_ONLY, threshold=threshold, source_views=count,
          prior_run=parent.RUN, predictor_floats=sum(x.numel() for group in design for x in group),
          seconds=perf_counter()-started))


def gate(root):
    info = read(root/'calibration.json')
    if any(info.get(key)!=value for key,value in SOURCE_ONLY.items()):
        raise ValueError('Expected the fixed source-only predictor calibration')
    return info['threshold']


def load(root, contract, threshold):
    saved = torch.load(root/'calibration.pt',map_location=DEVICE,weights_only=True)
    if (saved['contract'],saved['threshold']) != (contract,threshold):
        raise ValueError('Expected the same source and threshold contract')
    return parent.load(root.parent/parent.RUN,contract,threshold), saved['design']


def common(x, physical_rms):
    weight = physical_rms.square()
    return (x*weight).sum(-2,keepdim=True)/weight.sum(-2,keepdim=True).clamp_min(1e-8)


def common_directions(y, rms, model):
    prior, design = model
    anchor, _ = tracked_anchor(prior,y,rms)
    fixed_common = common(anchor,rms).expand_as(anchor)
    fixed_private = anchor - fixed_common
    desired = desired_bins(y.device)
    bands = band_masks(y.device)[:2]
    heard = common(y-y.mean(-1,keepdim=True),rms)
    tone = y.mean(-1,keepdim=True).abs()
    edits = [anchor]
    for features,(mean,center,coeff) in zip(input_features(y,desired),design,strict=True):
        values = torch.einsum('bfki,fkio->bfko',features-mean,coeff) + center
        complex_prediction = torch.view_as_complex(values.unflatten(-1,(2,2)).contiguous())
        z = complex_prediction.permute(0,3,1,2).flatten(1,2)
        spectrum = y.new_zeros(y.shape)
        spectrum[...,desired] = z
        proposed = .5*(common(ifft(spectrum)*tone,rms) - heard)
        moved = limited_turn(fixed_common, proposed.expand_as(y),rms,bands)
        edits.append(fixed_private + moved)
    return torch.stack(edits)


def recover(iq, tracker, model, threshold):
    y, rotation, _, rms = coordinates(tracker,iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    out = iq.expand(len(MODES)-2,*iq.shape).clone()
    evidence = iq.new_zeros((len(iq),len(FIELDS)))
    if count:
        y, rms = y[active],rms[active]
        updates = common_directions(y,rms,model)
        shared = common(updates,rms)
        error = ((updates-shared)-(updates[:1]-shared[:1])).abs().amax((0,2,3))
        doses = physical_norm(updates,rms).reshape(len(updates),-1)
        drift = (doses-doses[:1]).abs().amax(0)
        if not torch.isfinite(updates).all() or error.max()>1e-5 or drift.max()>1e-5:
            raise ValueError('Expected fixed private edit and equal physical RMS')
        out_update = updates*rotation[active]*rms
        out[:,active] = iq[active] + torch.stack((out_update.real,out_update.imag),dim=-2)
        size = physical_norm(y,rms).flatten().clamp_min(1e-8)
        evidence[active] = torch.stack((torch.ones_like(size),doses[0]/size,doses[2]/size,error,drift),-1)
    return out,count,evidence


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None):
    reference,_ = baseline(iq)
    out,count,measures = recover(iq,tracker,models[0],threshold)
    if measurements is not None:
        measurements.append(np.asarray(measures.tolist()))
    return torch.cat((torch.stack((iq,reference)),out)),count


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,modes=MODES,comparisons=COMPARISONS,fit=calibrate,load_gate=gate,
         check=partial(parent.smoke,modes=MODES,loader=load,renderer=render,recovery=recover),
         evaluate=partial(parent.evaluate,modes=MODES,fields=FIELDS,loader=load,renderer=render))
