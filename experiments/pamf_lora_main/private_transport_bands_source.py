"""Evaluate all fixed three-band private Wiener masks on the two LoRa routes."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .private_transport_bands import SUBSETS, band_deltas, named_band_masks
from .private_transport_source import dispatch, proposed_reads, readouts as parent_reads, timing
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-private-band-factorial-source-m2-r01'
MODES = ('direct','reference','rank_half',*(f'bands_{x}' for x in SUBSETS))
PAIRS = (*((f'bands_{x}', 'bands_000') for x in SUBSETS[1:]),
         *((f'bands_{x}', 'bands_111') for x in SUBSETS[1:-1]),
         ('bands_000','rank_half'))


@torch.no_grad()
def readouts(iq,tracker,model,threshold):
    y,rotation,q,rms=coordinates(tracker,iq)
    active=corruption_score(y)>threshold
    n=int(active.sum())
    if not n:
        return [iq]*(len(MODES)-2),0
    y,rotation,q,rms=(v[active] for v in (y,rotation,q,rms))
    rank,deltas=proposed_reads(y,q,model)
    bands=band_deltas(deltas[0],deltas[2],named_band_masks(y))
    base=iq.clone()
    base[active]+=real_iq((rank-y)*rotation*rms)
    decoded=[base]
    for edit in bands.values():
        output=base.clone()
        output[active]+=real_iq(edit*rotation*rms)
        decoded.append(output)
    return decoded,n


@torch.no_grad()
def render(iq,_unused,baseline,tracker,models,*,threshold):
    reference,_=baseline(iq)
    recovered,n=readouts(iq,tracker,models[0],threshold)
    return torch.stack((iq,reference,*recovered)),n


def evaluate(rt,route):
    folder,device=Path(rt['run_root']),'cuda:0'
    binding=source_control(rt,route,device)
    model,threshold=parent_model(folder,binding[2],device)
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=threshold),
          binding=binding,frozen_models=(model,),require_clean_bypass=False,
          replay_fn=receive_only,finalize=freeze,reader_chunk=144)


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
        raise ValueError('The actual source IQ and frozen readers differ by contract')
    model,threshold=parent_model(root,contract,device)
    rng=torch.Generator(device=device).manual_seed(2026113627)
    iq=real_iq(colored_observation(complex_iq(cal[:3]),rng))
    observed,n=render(iq,None,baseline,tracker,(model,),threshold=-1.)
    old,_=parent_reads(iq,tracker,model,-1.)
    for label,previous in (('rank_half',0),('bands_000',1),('bands_111',3)):
        if not torch.equal(observed[MODES.index(label)],old[previous]):
            raise ValueError(f'Source no-query smoke changed the {label} endpoint')
    classes=load_reader(root,contract,device)(observed.flatten(0,1))[0]
    if n!=len(iq) or classes.shape!=(len(MODES)*len(iq),10) or not torch.isfinite(classes).all():
        raise ValueError('Real LoRa band-factorial score smoke failed')
    write(root/'smoke.json',dict(status='PASS',source_u=len(iq),threshold=threshold,
          unchanged_parent_rows=['rank_half','bands_000','bands_111'],target_reads=0))
    print('LORA_BAND_FACTORIAL_REAL_CHECKPOINT_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    pub=commands.add_parser('publish')
    for key in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','timing','evaluate','score'):
        entry=commands.add_parser(action)
        entry.add_argument('runtime',type=Path)
        if action!='dispatch':
            entry.add_argument('row',choices=(*ROUTES,'reserve') if action=='evaluate' else ('source',))
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
            timing(rt,reader=readouts,timing_scope='one frozen parent and eight masks; excludes DURA & classifier')
        elif options['row']=='reserve':
            evaluate_reserve(rt)
        else:
            evaluate(rt,options['row'])


if __name__=='__main__':
    main()
