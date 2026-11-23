"""Measure per-band support, dose, and full-source clean retention for LoRa."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import (dispatch, evaluate_reserve, independent_score, limits,
                                    source_calibrate)
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import publish, score
from .data import VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .runner import activate
from .spectral_action import apply_bands, band_doses, band_support
from .spectral_unroll import ROUTES
from .weak_guard import generic_noise
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-rank-spectral-action-source-m2-r01'
SCHEMA = 'lora_band_action_calibration_v1'
CONTROLS = ('open_1_8', 'open_9_32', 'open_33_64', 'open_65_128')
CANDIDATES = ('band_half', 'band_taper', 'band_boost')
MODES = ('direct', 'reference', 'rank_old', *CONTROLS, *CANDIDATES)
PAIRS = (*((name, 'rank_old') for name in MODES[3:]), ('band_boost', 'band_taper'), ('band_half', 'band_taper'))


def spectral_bands(model):
    bands = model.bands.clone()
    bands[:, 0] = 0.
    return bands


def measure(model, observed, quality):
    return band_support(model.clean_detail, observed, quality, spectral_bands(model), model.selectors)[0]


def load_limits(root, device, original):
    return limits(root, device, original, schema=SCHEMA, count=4)


def edited(legacy, dft, addition, bands, rotation, rms):
    affected = (addition > 0).any(1)
    if not affected.any():
        return legacy
    edit = apply_bands(dft[affected], addition[affected], bands)
    delta = torch.fft.ifft(edit, norm='ortho')
    result = legacy.clone()
    result[affected] += real_iq(delta * rotation[affected] * rms[affected])
    return result


@torch.no_grad()
def recover(iq, tracker, model, original, threshold, *, telemetry=None):
    observed, rotation, q, rms = coordinates(tracker, iq)
    old = corruption_score(observed) > original
    bands = spectral_bands(model)
    gain, optimum = band_support(model.clean_detail, observed, q, bands, model.selectors)
    base = .5 * old[:, None].to(gain.dtype)
    hard, taper, boost = band_doses(old, gain, optimum, threshold)
    additions = (hard-base, taper-base, boost-base)
    legacy = iq.clone()
    dft = torch.empty_like(observed)
    for legacy_group, select in ((True, old), (False, ~old)):
        if select.any():
            original_wave = observed[select]
            proposal = model(original_wave, q[select], rotation[select])[-1]
            dft[select] = torch.fft.fft(proposal-original_wave,norm='ortho')
            if legacy_group:
                legacy[select] += real_iq((torch.lerp(original_wave,proposal,.5)-original_wave) *
                                          rotation[select] * rms[select])

    basis = torch.eye(4, device=iq.device, dtype=iq.dtype)
    always = 0.5 * (~old[:,None]).to(gain.dtype)
    controls = [edited(legacy,dft,always*one,bands,rotation,rms) for one in basis]
    candidates = [edited(legacy,dft,increment,bands,rotation,rms) for increment in additions]
    if telemetry is not None:
        telemetry.append(torch.cat((old[:,None].float(),*additions),dim=1).cpu().numpy())
    active = int((old|(additions[0]>0).any(1)).sum())
    return (legacy, *controls, *candidates), active


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, original, thresholds, telemetry=None):
    reference, _ = baseline(iq)
    candidates, active = recover(iq, tracker, models[0], original, thresholds, telemetry=telemetry)
    return torch.stack((iq, reference, *candidates)), active


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    rank, old = parent_model(root, binding[2], device)
    threshold = load_limits(root, device, old)
    telemetry = []
    renderer = partial(render, original=old, thresholds=threshold, telemetry=telemetry)
    score(rt, route, panel=panel, modes=MODES, render=renderer, binding=binding, frozen_models=(rank,),
          require_clean_bypass=False, replay_fn=receive_only, finalize=freeze, reader_chunk=144)
    data = np.concatenate(telemetry, axis=0).reshape(len(VIEWS), -1, 13)
    summary={}
    for name, values in zip(VIEWS, data, strict=True):
        old=values[:,0]>0
        change=values[:,1:].reshape(-1,3,4)
        summary[name]=dict(n=int(len(old)),old_active=int(old.sum()),
                           additional_packets=((change>0).any(2)&~old[:,None]).sum(0).tolist(),
                           active_band_counts=(change>0).sum(0).tolist(),
                           mean_additional_dose=change.mean(0).tolist(),
                           boosted_existing=(change[old]>0).sum(0).tolist())
    write(root/f'{route}-band-counts.json',dict(modes=list(CANDIDATES),by_view=summary))


@torch.no_grad()
def timing(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    model,original=parent_model(root,contract,device)
    threshold=load_limits(root,device,original)
    tracker=PacketRestorer('consensus').to(device).eval()
    bands=spectral_bands(model)
    rng=torch.Generator(device=device).manual_seed(2026113289)
    distorted=real_iq(generic_noise(complex_iq(cal[:1]),rng,amplitude=.12))
    timings={}
    for key,iq in (('clean',cal[:1]),('generic',distorted)):
        elapsed=[]
        for _ in range(32):
            torch.cuda.synchronize()
            start=perf_counter()
            y,phase,q,rms=coordinates(tracker,iq)
            old=corruption_score(y)>original
            gain,optimal=band_support(model.clean_detail,y,q,bands,model.selectors)
            boost=band_doses(old,gain,optimal,threshold)[-1]
            active=boost.amax().item()>0
            result=iq
            if active:
                corrected=model(y,q,phase)[-1]
                base=.5 * old[:,None].to(boost.dtype)
                if bool(old[0]):
                    result=iq+real_iq((torch.lerp(y,corrected,.5)-y)*phase*rms)
                extra=boost-base
                if (extra>0).any():
                    fourier=torch.fft.fft(corrected-y,norm='ortho')
                    result=edited(result,fourier,extra,bands,phase,rms)
            torch.cuda.synchronize()
            elapsed.append((perf_counter()-start)*1000)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite LoRa timed band output')
        timings[key]=dict(median_ms=float(np.median(elapsed[8:])),active=int(active))
    write(root/'b1.json',dict(scope='band_boost single readout before frozen classifier',timing=timings,
           gpu=torch.cuda.get_device_name(),n_measurements=24,parameters=sum(p.numel() for p in model.parameters())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    entry=commands.add_parser('publish')
    for option in ('repo','output','context','ssh-config','parent-submit'):
        entry.add_argument('--'+option,type=Path,required=True)
    for stage in ('dispatch','smoke','fit','timing','evaluate','score'):
        entry=commands.add_parser(stage)
        entry.add_argument('runtime',type=Path)
        if stage!='dispatch':
            entry.add_argument('row',choices=(*ROUTES,'reserve') if stage=='evaluate' else ('source',))
    opts=vars(parser.parse_args())
    action=opts.pop('action')
    if action=='publish':
        publish(**opts,run=RUN,runner=__spec__.name)
        return
    rt=read(opts['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        independent_score(rt,modes=MODES,comparisons=PAIRS)
    else:
        activate(rt)
        if action=='evaluate':
            if opts['row']=='reserve':
                evaluate_reserve(rt,modes=MODES,renderer=render,load_limits=load_limits)
            else:
                evaluate(rt,opts['row'])
        elif action=='timing':
            timing(rt)
        else:
            source_calibrate(rt,smoke=action=='smoke',renderer=render,measure=measure,
                             modes=MODES,schema=SCHEMA,smoke_edit_index=3)


if __name__=='__main__':
    main()
