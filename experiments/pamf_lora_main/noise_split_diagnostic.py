"""Isolated LoRa attribution of known additive noise to packet and spectral parts."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind_source import panel, receive_only, setup
from .complex_vector_source import load_reader, score as predict
from .data import SOURCE, VIEWS
from .denoise_source import replay
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, publish, score_frozen
from .noise_floor_diagnostic import BANDS, ROUTES, dispatch
from .runner import activate


RUN = '20261124-diagnostic-lora-noise-split-source-m2-r01'
BAND_MODES = ('exact_1_8','exact_9_32','exact_33_64','exact_65_128')
MODES = ('direct','reference','exact_common','exact_private','exact_noncarrier','exact_full',
         *BAND_MODES,'noncarrier_phase','noncarrier_amplitude')
FACTORS = (('exact_noncarrier','exact_common'),('exact_noncarrier','exact_private'),
           ('exact_full','exact_noncarrier'),('exact_noncarrier','noncarrier_phase'),
           ('exact_noncarrier','noncarrier_amplitude'),('exact_common','exact_private'))
COMPARISONS = (*FACTORS, *((mode,base) for mode in MODES[2:] for base in ('direct','reference')))
DEVICE = 'cuda:0'


def components(exact):
    spec = torch.fft.fft(exact, norm='ortho')
    f = torch.arange(spec.shape[-1], device=spec.device)
    distance = torch.minimum(f, spec.shape[-1]-f)
    noncarrier = spec * (distance>0)
    common = noncarrier.mean(dim=1, keepdim=True).expand_as(noncarrier)
    bands = [spec*((distance>=lo)&(distance<=hi)) for lo,hi in BANDS]
    return common,noncarrier-common,noncarrier,*bands


def exact_readouts(iq, epsilon, tracker):
    y = complex_iq(iq)
    truth = complex_iq(epsilon)
    active = epsilon.abs().amax((1,2,3))>0.
    result = iq.expand(len(MODES)-2,*iq.shape).clone()
    levels = iq.new_zeros((len(iq),9))
    if not active.any():
        return result, int(active.sum()), levels
    selected = iq[active]
    _, rotation, _, rms = coordinates(tracker,selected)
    frame = rotation*rms
    known = truth[active]
    parts = components(known/frame)
    corrections = torch.fft.ifft(torch.stack(parts),norm='ortho')*frame
    common,private,noncarrier,*bands = corrections
    raw = y[active]
    cleaned = raw-noncarrier
    phase = torch.polar(raw.abs(),cleaned.angle())
    amplitude = torch.polar(cleaned.abs(),raw.angle())
    waves = (raw-common,raw-private,cleaned,raw-known,*(raw-band for band in bands),phase,amplitude)
    prepared = torch.stack(waves)
    result[:,active] = torch.stack((prepared.real,prepared.imag),dim=-2)
    energy = known.abs().square().sum((1,2)).clamp_min(1e-10)
    correction_power = corrections.abs().square().sum((-1,-2)).mT / energy[:,None]
    levels[active] = torch.cat((torch.ones_like(energy[:,None]),
                                energy[:,None]/raw.abs().square().sum((1,2)).clamp_min(1e-10)[:,None],
                                correction_power),dim=1)
    return result,int(active.sum()),levels


def replay_trace(route, view, row):
    measured,noise = replay(route,view,row)
    if route=='residual' and not np.array_equal(measured,receive_only(route,view,row)[0]):
        raise ValueError('Isolated noise trace must reproduce the same registered residual IQ')
    return measured,noise


def render(iq, epsilon, baseline, tracker, _models, *, measurements=None):
    ref,_ = baseline(iq)
    waves,active,detail = exact_readouts(iq,epsilon,tracker)
    if measurements is not None:
        measurements.append(np.asarray(detail.tolist()))
    return torch.cat((torch.stack((iq,ref)),waves)),active


@torch.no_grad()
def smoke(rt):
    (_,u,_),contract = setup(rt,DEVICE,smoke=True)
    tracker = PacketRestorer('consensus').to(DEVICE).eval()
    reader = load_reader(Path(rt['run_root']),contract,DEVICE)
    clean = u[:4]
    rng = torch.Generator(device=DEVICE).manual_seed(2026112491)
    noise = .3*torch.randn((2,6,2,256),device=DEVICE,generator=rng)
    eps = torch.cat((torch.zeros_like(noise),noise))
    raw = clean+eps
    waves,active,_ = exact_readouts(raw,eps,tracker)
    torch.testing.assert_close(waves[3],clean,atol=3e-7,rtol=1e-5)
    if not torch.equal(waves[:,:2],clean[:2].expand_as(waves[:,:2])) or active!=2:
        raise ValueError('Exact decomposition must preserve zero-noise observations')
    logits = reader(waves.flatten(0,1))[0]
    if logits.shape != (4*(len(MODES)-2),10) or not torch.isfinite(logits).all():
        raise ValueError('Expected real reader outputs in diagnostic smoke')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',diagnostic_only=True,target_reads=0,V_reads=0))


def evaluate(rt,route):
    binding = source_control(rt,route,DEVICE)
    evidence = []

    def finalize(root, route, source, chosen, ids, predictions, metrics, modes):
        array = np.concatenate(evidence).reshape(len(VIEWS),len(chosen),9)
        result = {}
        for view, cohort in zip(VIEWS,array,strict=True):
            per_rx = {}
            for rx, group in zip(SOURCE,np.split(cohort,len(SOURCE),axis=0),strict=True):
                active = group[group[:,0]>0.]
                per_rx[rx] = dict(count=len(active),median=np.median(active[:,1:],axis=0).tolist() if len(active) else [])
            result[view] = per_rx
        order = ('noise_over_observed','common','private','noncarrier',*BAND_MODES)
        write(root/f'{route}-isolated-estimates.json',dict(diagnostic_only=True,fields=order,by_view=result))
        freeze(root,route,source,chosen,ids,predictions,metrics,modes)

    predict(rt,route,panel=panel,modes=MODES,binding=binding,frozen_models=(),replay_fn=replay_trace,
            render=partial(render,measurements=evidence),require_clean_bypass=True,finalize=finalize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','evaluate','score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime',type=Path)
        if action!='dispatch':
            worker.add_argument('name',choices=ROUTES if action=='evaluate' else ('source',))
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
        if action=='smoke':
            smoke(rt)
        else:
            evaluate(rt,args['name'])


if __name__ == '__main__':
    main()
