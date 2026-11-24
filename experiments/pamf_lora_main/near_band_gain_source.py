"""Calibrate a small observed-only gain for near-carrier LoRa private detail."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_band_power_source import model_binding
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .crossfit_pilot import apply_transfer, generic_taps
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_mode_covariance_source import dispatch
from .joint_unroll_source import coordinates, source_control
from .near_band_gain import GAIN_CAP, adjust, fit_coefficients, increase, inputs, protected_target
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-near-band-amplitude-source-m2-r01'
SEED = 2026115323
MODES = ('direct', 'reference', 'antithetic_half', 'fixed_gain', 'observed_gain', 'parity_gain')
PAIRS = (('parity_gain','observed_gain'), ('parity_gain','antithetic_half'),
         ('observed_gain','antithetic_half'), ('fixed_gain','antithetic_half'))


def project(frame, model):
    observed, rot, quality, rms = frame
    _, full = model(observed, quality, rot)
    half = torch.lerp(observed, full, .5)
    return observed, half, rot * rms


def calibrate(packets, tracker, model, threshold, seed):
    rng = torch.Generator(device=packets.device).manual_seed(seed)
    features, goals = [], []
    for iq in packets.split(32):
        clean = complex_iq(iq)
        taps = generic_taps(clean, rng)
        taps *= (torch.rand((len(clean),1), device=clean.device, generator=rng) < .5)
        affected = apply_transfer(clean, taps)
        received = real_iq(colored_observation(affected, rng))
        frame = coordinates(tracker, received)
        mask = corruption_score(frame[0]) > threshold
        if not mask.any():
            continue
        observed, half, position = project(tuple(v[mask] for v in frame), model)
        vector, output, basis, bins = inputs(observed, half)
        features.append(vector)
        goals.append(protected_target(output, clean[mask] / position, basis, bins))
    return torch.cat(features), torch.cat(goals)


@torch.no_grad()
def fit(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, _, held), contract = setup(rt, device)
    anchor, threshold = model_binding(rt, contract, device=device)
    tracker = PacketRestorer('consensus').to(device).eval()
    started = perf_counter()
    features, target = calibrate(training, tracker, anchor, threshold, SEED)
    fits = [fit_coefficients(features[:, :width], target) for width in (4,5)]
    valid, keys = calibrate(held, tracker, anchor, threshold, SEED+1)
    decoded = [increase(valid[:, :len(arm[0])], arm) for arm in fits]
    if not all(v.isfinite().all() for v in (*decoded, keys)):
        raise ValueError('Nonfinite calibration gain')
    with (root/'gain-coefficients.pt').open('xb') as stream:
        torch.save(dict(contract=contract, coeffs=[tuple(x.cpu() for x in t) for t in fits]), stream)
    write(root/'fit.json',dict(status='SOURCE_FROZEN',source_U=len(training),fit_active=len(target),
          held_U=len(held),held_active=len(keys),held_target_mean=float(keys.mean()),
          generic_held_mse=[float((x-keys).square().mean()) for x in decoded],
          generic_held_gain_mean=[float(x.mean()) for x in decoded], seconds=perf_counter()-started,
          trainable_parameters=0,target_reads=0,F_reads=0,TX_reads=0,V_reads=0))


def bound(root, contract, device):
    model, threshold = model_binding(dict(run_root=str(root)), contract, device=device)
    if read(root/'fit.json')['status'] != 'SOURCE_FROZEN':
        raise ValueError('Expected frozen gain calibration')
    values = torch.load(root/'gain-coefficients.pt', map_location=device, weights_only=True)
    if values['contract'] != contract:
        raise ValueError('Gain calibration source differs from inference source')
    return model,threshold,values['coeffs']


@torch.no_grad()
def readouts(iq, tracker, model, threshold, coefficients):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    count = int(active.sum())
    if not count:
        return [iq] * (len(MODES)-2), 0
    y, half, position = project(tuple(v[active] for v in frame), model)
    features, output, basis, bins = inputs(y, half)
    gains = (y.real.new_full((count,), GAIN_CAP/2),
             *(increase(features[:, :len(theta[0])], theta) for theta in coefficients))
    rows = []
    for value in (half, *(adjust(half, output, basis, bins, gain) for gain in gains)):
        current = iq.clone()
        current[active] += real_iq((value-y)*position)
        rows.append(current)
    return rows, count


@torch.no_grad()
def render(iq, _, baseline, tracker, models):
    ref, _ = baseline(iq)
    rows, active = readouts(iq, tracker, *models)
    return torch.stack((iq, ref, *rows)), active


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    score(rt, route, panel=panel, modes=MODES, render=render, binding=binding,
          frozen_models=bound(root,binding[2],device), replay_fn=receive_only,
          finalize=freeze, reader_chunk=144, require_clean_bypass=False)


def reserve(rt):
    def bind(root, contract, device, baseline, tracker):
        models=bound(root,contract,device)
        return lambda iq: render(iq,None,baseline,tracker,models)
    predict_clean_reserve(rt,modes=MODES,bind=bind)


@torch.no_grad()
def smoke(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (training,cal,_),contract=setup(rt,device,smoke=True)
    _,_,matched,baseline,tracker=source_control(rt,'full',device)
    if matched!=contract:
        raise ValueError('Smoke source contract mismatch')
    model,threshold=model_binding(rt,contract,device=device)
    features,target=calibrate(training,tracker,model,threshold,SEED)
    coefficients=[fit_coefficients(features[:, :k],target) for k in (4,5)]
    iq=real_iq(colored_observation(complex_iq(cal),torch.Generator(device=device).manual_seed(SEED)))
    touched=corruption_score(coordinates(tracker,iq)[0])>threshold
    tested=torch.cat((cal[:2],iq[touched][:2]))
    waves,count=render(tested,None,baseline,tracker,(model,threshold,coefficients))
    logits=load_reader(root,contract,device)(waves.flatten(0,1))[0]
    if (count!=2 or logits.shape!=(len(tested)*len(MODES),10) or not logits.isfinite().all()
            or not torch.equal(waves[2:,:2],cal[:2].expand_as(waves[2:,:2]))
            or torch.equal(waves[2,2:],waves[3,2:])):
        raise ValueError('Real no-query near-band amplitude smoke failed')
    write(root/'smoke.json',dict(status='PASS',source_U=len(tested),active=count,target_reads=0))
    print('LORA_NEAR_BAND_GAIN_REAL_CHECKPOINT_PASS',flush=True)


@torch.no_grad()
def timing(rt):
    device='cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    models=bound(Path(rt['run_root']),contract,device)
    tracker=PacketRestorer('consensus').to(device).eval()
    iq=real_iq(colored_observation(complex_iq(cal),torch.Generator(device=device).manual_seed(SEED)))
    strongest=corruption_score(coordinates(tracker,iq)[0]).argmax().view(1)
    periods={}
    for name,data in (('clean', cal[:1]),('generic',iq[strongest])):
        elapsed=[]
        for _ in range(32):
            torch.cuda.synchronize()
            started=perf_counter()
            readouts(data,tracker,*models)
            torch.cuda.synchronize()
            elapsed.append(1000*(perf_counter()-started))
        periods[name]=float(np.median(elapsed[8:]))
    write(Path(rt['run_root'])/'b1.json',dict(gpu=torch.cuda.get_device_name(),median_ms=periods,
          n_measurements=24,scope='one anchor inference and three correction readouts; excludes DURA and F'))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    commands=cli.add_subparsers(dest='action',required=True)
    pub=commands.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for phase in ('dispatch','smoke','fit','timing','evaluate','score'):
        runner=commands.add_parser(phase)
        runner.add_argument('runtime',type=Path)
        if phase!='dispatch':
            runner.add_argument('row',choices=(*ROUTES,'reserve') if phase=='evaluate' else ('source',))
    args=vars(cli.parse_args())
    phase=args.pop('action')
    if phase=='publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt=read(args['runtime'])
    if phase=='dispatch':
        return dispatch(rt)
    if phase=='score':
        return independent_score(rt,modes=MODES,comparisons=PAIRS)
    activate(rt)
    if phase=='smoke':
        return smoke(rt)
    if phase=='fit':
        return fit(rt)
    if phase=='timing':
        return timing(rt)
    return reserve(rt) if args['row']=='reserve' else evaluate(rt,args['row'])


if __name__=='__main__':
    main()
