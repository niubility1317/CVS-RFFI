"""Compare two private-spectrum gains on one frozen LoRa rank-four recovery."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score, zero_carrier
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .private_transport import private_modes, replace_private
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .variance_bound import spectral_contrast, variance_interval
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-private-variance-transport-source-m2-r01'
MODES = ('direct','reference','rank_half','wiener_half','transport_half','wiener_full','transport_full')
PAIRS = (('transport_half','wiener_half'),('transport_full','wiener_full'),
         ('wiener_half','rank_half'),('transport_half','rank_half'),
         ('wiener_full','rank_half'),('transport_full','rank_half'))


def proposed_reads(observed, q, model):
    """Return the unmodified rank midpoint and four matched private readouts."""
    _,proposal,outer = model.propose(observed,q)
    spectrum = torch.fft.fft(observed,norm='ortho')
    center = torch.fft.fft(proposal,norm='ortho')
    contrast, weight = spectral_contrast(spectrum,q)
    band_noise,_ = variance_interval(outer,contrast,weight,model.selectors)
    noise = band_noise @ model.bands
    corrected = center + model.clean_detail.innovation(spectrum-center,noise,True)
    refined = observed + zero_carrier(torch.fft.ifft(corrected-spectrum,norm='ortho'))
    rank = torch.lerp(observed,refined,.5)
    prior = model.clean_detail
    bins = prior.correlated_bins
    old = torch.fft.fft(rank,norm='ortho')
    wiener,transport = private_modes(spectrum,center,band_noise,model.bands,prior)
    received = spectrum.index_select(-1,bins)
    return rank, tuple(torch.fft.ifft(replace_private(old,bins,row)-old,norm='ortho')
                       for row in (torch.lerp(received,wiener,.5),torch.lerp(received,transport,.5),
                                   wiener,transport))


@torch.no_grad()
def readouts(iq, tracker, model, threshold):
    y, rot, quality, rms=coordinates(tracker,iq)
    active=corruption_score(y)>threshold
    if not active.any():
        return [iq]*(len(MODES)-2),0
    y,rot,q,rms=(v[active] for v in (y,rot,quality,rms))
    midpoint, changes=proposed_reads(y,q,model)
    reference=iq.clone()
    reference[active]+=real_iq((midpoint-y)*rot*rms)
    rows=[reference]
    for change in changes:
        out=reference.clone()
        out[active]+=real_iq(change*rot*rms)
        rows.append(out)
    return rows,int(active.sum())


@torch.no_grad()
def render(iq,_unused,baseline,tracker,models,*,threshold):
    reference,_=baseline(iq)
    waves,n=readouts(iq,tracker,models[0],threshold)
    return torch.stack((iq,reference,*waves)),n


def evaluate(rt,route):
    root,device=Path(rt['run_root']),'cuda:0'
    binding=source_control(rt,route,device)
    model,threshold=parent_model(root,binding[2],device)
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=threshold),binding=binding,
          frozen_models=(model,),require_clean_bypass=False,replay_fn=receive_only,finalize=freeze,reader_chunk=144)


def evaluate_reserve(rt):
    def bind(root,contract,device,baseline,tracker):
        model,threshold=parent_model(root,contract,device)
        return partial(render,_unused=None,baseline=baseline,tracker=tracker,models=(model,),threshold=threshold)
    predict_clean_reserve(rt,modes=MODES,bind=bind)


@torch.no_grad()
def smoke(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    _,_,expected,baseline,tracker=source_control(rt,'full',device)
    if expected!=contract:
        raise ValueError('Actual source U and frozen reader contract differ')
    model,threshold=parent_model(root,contract,device)
    gen=torch.Generator(device=device).manual_seed(2026113627)
    iq=real_iq(colored_observation(complex_iq(cal[:3]),gen))
    y,phase,q,_=coordinates(tracker,iq)
    base,deltas=proposed_reads(y,q,model)
    expected=torch.lerp(y,model(y,q,phase)[-1],.5)
    if not torch.equal(base,expected):
        raise ValueError('Rank parent changed during the private-only readout')
    avg=max(float(d.mean(1).abs().amax()) for d in deltas)
    if avg>2e-6 or not all(torch.isfinite(d).all() for d in deltas):
        raise ValueError('Private update changed the common spectrum')
    stack,n=render(iq,None,baseline,tracker,(model,),threshold=-1.)
    reader=load_reader(root,contract,device)
    logits=reader(stack.flatten(0,1))[0]
    if n!=len(iq) or logits.shape!=(len(iq)*len(MODES),10) or not torch.isfinite(logits).all():
        raise ValueError('Frozen private transport smoke failed')
    write(root/'smoke.json',dict(status='PASS',source_u=len(iq),threshold=threshold,
          common_drift_max=avg,target_reads=0))
    print('LORA_PRIVATE_TRANSPORT_REAL_CHECKPOINT_PASS',flush=True)


@torch.no_grad()
def timing(rt, *, reader=readouts, timing_scope=None):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    model,threshold=parent_model(root,contract,device)
    tracker=PacketRestorer('consensus').to(device).eval()
    generator=torch.Generator(device=device).manual_seed(2026113627)
    contaminated=real_iq(colored_observation(complex_iq(cal),generator))
    pick=corruption_score(coordinates(tracker,contaminated)[0]).argmax().view(1)
    periods={}
    for label,iq in (('clean',cal[:1]),('generic',contaminated[pick])):
        elapsed=[]
        for _ in range(32):
            torch.cuda.synchronize()
            t0=perf_counter()
            _,n=reader(iq,tracker,model,threshold)
            torch.cuda.synchronize()
            elapsed.append(1000*(perf_counter()-t0))
        periods[label]=dict(active=n,median_ms=float(np.median(elapsed[8:])))
    write(root/'b1.json',dict(gpu=torch.cuda.get_device_name(),n_measurements=24,timing=periods,
          restoration_weights=sum(v.numel() for v in model.parameters()),
          scope=timing_scope or 'shared rank parent and four private engineering readouts; excludes classifier and DURA'))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    execute=partial(start,module=rt['runner'])
    try:
        for phase,rows in (('smoke',('source',)),('timing',('source',)),
                           ('evaluate',ROUTES),('evaluate',('reserve',))):
            schedule(rt,[(phase,row) for row in rows],execute)
        if execute(rt,'score','source').wait():
            raise RuntimeError('Independent private transport scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as e:
        write(root/'state.json',dict(status='FAILED',error=str(e)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    pub=commands.add_parser('publish')
    for key in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','timing','evaluate','score'):
        stage=commands.add_parser(action)
        stage.add_argument('runtime',type=Path)
        if action!='dispatch':
            stage.add_argument('row',choices=(*ROUTES,'reserve') if action=='evaluate' else ('source',))
    options=vars(parser.parse_args())
    action=options.pop('action')
    if action=='publish':
        publish(**options,run=RUN,runner=__spec__.name)
        return
    rt=read(options['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        independent_score(rt,modes=MODES,comparisons=PAIRS)
    else:
        activate(rt)
        if action=='smoke':
            smoke(rt)
        elif action=='timing':
            timing(rt)
        elif options['row']=='reserve':
            evaluate_reserve(rt)
        else:
            evaluate(rt,options['row'])


if __name__=='__main__':
    main()
