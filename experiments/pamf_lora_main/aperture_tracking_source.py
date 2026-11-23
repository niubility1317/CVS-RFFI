"""Evaluate a receive-only LoRa timing and aperture engineering control."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .aperture_tracking import ApertureTracking, carrier_from_frame
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener import LocalWiener
from .local_wiener_source import ROUTES, dispatch, freeze, publish, score_frozen, threshold
from .runner import activate
from .spectral_block_recovery import varied_observation


RUN = '20261124-phase1-lora-aperture-tracking-source-m2-r01'
MODES = ('direct','reference','legacy_relative','split_relative','adaptive_relative','adaptive_half',
         'legacy_wiener','split_wiener','adaptive_wiener')
COMPARISONS = (('split_relative','legacy_relative'),('adaptive_relative','split_relative'),
               ('split_wiener','legacy_wiener'),('adaptive_wiener','legacy_wiener'),
               ('direct','reference'),
               *((name,base) for name in MODES[2:] for base in ('reference','direct')))


@torch.no_grad()
def calibrate(rt):
    started = perf_counter()
    device = 'cuda:0'
    (_, source, _), _ = setup(rt,device,fit_per_rx=4000)
    tracker = PacketRestorer('consensus').to(device).eval()
    levels = torch.cat([corruption_score(coordinates(tracker,iq)[0]) for iq in source.split(48)])
    if len(levels)!=2500 or not torch.isfinite(levels).all():
        raise ValueError('Expected 2500 finite U-only packet measurements')
    write(Path(rt['run_root'])/'threshold.json',dict(threshold=float(1.05*levels.max()),rows=len(levels),
          F_reads=0,V_reads=0,TX_reads=0,target_reads=0,optimizer_updates=0,parameters=0,
          wall_seconds=perf_counter()-started))


def recover(iq, tracker, aperture, wiener, gate):
    measured, phase, _, rms = coordinates(tracker,iq)
    mask = corruption_score(measured)>gate
    output = [iq.clone() for _ in MODES[2:]]
    active = int(mask.sum())
    if active:
        y, received = measured[mask],complex_iq(iq[mask])
        chirp = tracker.reference.chirp
        legacy = carrier_from_frame(phase[mask],chirp)
        split, adaptive, _, _ = aperture.frequencies(iq[mask],chirp,legacy)
        changes = [real_iq(aperture.translate(received,x)) for x in (legacy,split,adaptive)]
        changes.append(real_iq(aperture.translate(received,adaptive,.5)))
        # Recompute the signal statistics in each frame. The scalar power
        # estimate, Wiener rule, and restoration dose stay matched.
        spectrum, level, _ = wiener.analyze(y)
        delta = wiener.restore(y,spectrum,level,correlated=True)*phase[mask]*rms[mask]
        changes.append(real_iq(received+.5*delta))
        for frequency in (split,adaptive):
            framed, frame = aperture.frame(received,frequency,chirp)
            spectrum = wiener.analyze(framed)[0]
            delta = wiener.restore(framed,spectrum,level,correlated=True)*frame
            changes.append(real_iq(received+.5*delta))
        for result, value in zip(output,changes,strict=True):
            result[mask] = value
    return output,active


def render(iq, _unused, reference, tracker, models, *, gate):
    baseline, _ = reference(iq)
    waves, active = recover(iq,tracker,*models,gate)
    return torch.stack((iq,baseline,*waves)),active


def instruments(device):
    return ApertureTracking().to(device).eval(), LocalWiener().to(device).eval()


@torch.no_grad()
def smoke(rt, gate):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    (_, source, _), contract = setup(rt,device,smoke=True)
    models = instruments(device)
    _, _, expected, baseline, tracker = source_control(rt,'full',device)
    if contract!=expected:
        raise ValueError('Clean source smoke and DURA require the same data contract')
    received = real_iq(varied_observation(complex_iq(source),torch.Generator(device=device).manual_seed(2026112403)))
    ratios = corruption_score(coordinates(tracker,received)[0])
    corrupted = received[ratios.argmax().view(1)]
    waves, active = render(torch.cat((source[:1],corrupted)),None,baseline,tracker,models,gate=gate)
    f = load_reader(root,contract,device)
    logits = f(waves.flatten(0,1))[0]
    if not active or logits.shape!=(2*len(MODES),10) or not torch.isfinite(logits).all():
        raise ValueError('Expected the real no-query source smoke to exercise all readouts')
    times = {}
    for label, iq in (('source_clean',source[:1]),('generic_disturbed',corrupted)):
        ms = []
        for i in range(28):
            torch.cuda.synchronize()
            begun = perf_counter()
            _, enabled = recover(iq,tracker,*models,gate)
            torch.cuda.synchronize()
            if i>=8:
                ms.append(1000*(perf_counter()-begun))
        times[label] = dict(active=bool(enabled),median_ms=float(np.median(ms)))
    write(root/'smoke.json',dict(status='PASS',target_reads=0,V_reads=0,parameters=0,
                                b1=times,device=torch.cuda.get_device_name()))


def evaluate(rt, route, gate):
    device = 'cuda:0'
    score(rt,route,panel=panel,modes=MODES,render=partial(render,gate=gate),
          binding=source_control(rt,route,device),frozen_models=instruments(device),
          require_clean_bypass=False,replay_fn=receive_only,finalize=freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','calibrate','smoke','evaluate','score'):
        sub = actions.add_parser(action)
        sub.add_argument('runtime',type=Path)
        if action!='dispatch':
            sub.add_argument('name',choices=ROUTES if action=='evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args,run=RUN,runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        score_frozen(rt,MODES,comparisons=COMPARISONS)
    else:
        activate(rt)
        if action=='calibrate':
            calibrate(rt)
        elif action=='smoke':
            smoke(rt,threshold(Path(rt['run_root'])))
        else:
            evaluate(rt,args['name'],threshold(Path(rt['run_root'])))


if __name__=='__main__':
    main()
