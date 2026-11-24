"""Test global and local cross-chirp subspaces and an equal-dose restoration guard."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read,write
from .action_support_source import independent_score,predict_clean_reserve
from .channel_blind import corruption_score,zero_carrier
from .channel_blind_source import panel,receive_only,setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader,publish,score
from .cross_chirp_hankel import LOCAL_HOP,LOCAL_WIDTH,apply_subspace,match_dose,subspace_filter
from .dura_repair import complex_iq,real_iq
from .joint_unroll_source import coordinates,source_control
from .private_transport_source import dispatch,timing
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_guard_source import parent_model
from .weak_signal_probe import observed_noise_rms


RUN = '20261124-phase1-lora-cross-chirp-hankel-source-m2-r01'
MODES = ('direct','reference','rank_half','global_hankel_half','local_hankel_half',
         'local_self_included_half','protected_hankel_half','matched_dose_half')
PAIRS = (('local_hankel_half','global_hankel_half'),('local_hankel_half','local_self_included_half'),
         ('protected_hankel_half','matched_dose_half'),('protected_hankel_half','rank_half'),
         ('matched_dose_half','rank_half'))


def waveform_edits(y,rot,q,model):
    # In each chirp the same observed noise proxy defines the unit noise floor.
    residual=zero_carrier(y)
    noise=observed_noise_rms(y)
    n=y.shape[-1]
    global_shape=subspace_filter(residual,noise,n,n,exclude_self=True)
    local_shape=subspace_filter(residual,noise,LOCAL_WIDTH,LOCAL_HOP,exclude_self=True)
    pooled=subspace_filter(residual,noise,LOCAL_WIDTH,LOCAL_HOP,exclude_self=False)
    broad=apply_subspace(residual,global_shape,n,n)
    narrow=apply_subspace(residual,local_shape,LOCAL_WIDTH,LOCAL_HOP)
    seen=apply_subspace(residual,pooled,LOCAL_WIDTH,LOCAL_HOP)
    midpoint=torch.lerp(y,model(y,q,rot)[-1],.5)
    parent=midpoint-y
    retained=apply_subspace(parent,local_shape,LOCAL_WIDTH,LOCAL_HOP)
    protected,dose=match_dose(parent,zero_carrier(parent-retained))
    return (parent,.5*zero_carrier(broad-residual),.5*zero_carrier(narrow-residual),
            .5*zero_carrier(seen-residual),protected,dose)


@torch.no_grad()
def readouts(iq,tracker,model,threshold):
    y,rot,q,rms=coordinates(tracker,iq)
    active=corruption_score(y)>threshold
    n=int(active.sum())
    if not n:
        return [iq]*(len(MODES)-2),0
    y,rot,q,rms=(v[active] for v in (y,rot,q,rms))
    modes=[]
    for dx in waveform_edits(y,rot,q,model):
        out=iq.clone()
        out[active]+=real_iq(dx*rot*rms)
        modes.append(out)
    return modes,n


@torch.no_grad()
def render(iq,_unused,baseline,tracker,models,*,threshold):
    reference,_=baseline(iq)
    waves,n=readouts(iq,tracker,models[0],threshold)
    return torch.stack((iq,reference,*waves)),n


def evaluate(rt,route):
    root,device=Path(rt['run_root']),'cuda:0'
    binding=source_control(rt,route,device)
    model,threshold=parent_model(root,binding[2],device)
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=threshold),
          binding=binding,frozen_models=(model,),require_clean_bypass=False,
          replay_fn=receive_only,finalize=freeze,reader_chunk=144)


def reserve(rt):
    def bind(root,contract,device,baseline,tracker):
        model,threshold=parent_model(root,contract,device)
        return partial(render,_unused=None,baseline=baseline,tracker=tracker,
                       models=(model,),threshold=threshold)
    predict_clean_reserve(rt,modes=MODES,bind=bind)


@torch.no_grad()
def smoke(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    _,_,expected,baseline,tracker=source_control(rt,'full',device)
    if expected!=contract:
        raise ValueError('Source observations and frozen reader have different contracts')
    model,threshold=parent_model(root,contract,device)
    rng=torch.Generator(device=device).manual_seed(2026114813)
    iq=real_iq(colored_observation(complex_iq(cal[:3]),rng))
    y,rot,q,_=coordinates(tracker,iq)
    edits=waveform_edits(y,rot,q,model)
    a,b=edits[-2:]
    torch.testing.assert_close(torch.linalg.vector_norm(a,dim=-1),
                               torch.linalg.vector_norm(b,dim=-1),atol=1e-5,rtol=1e-5)
    if max(float(dx.mean(-1).abs().max()) for dx in edits)>1e-6:
        raise ValueError('Hankel or matched-dose readout shifted the carrier')
    stacked,n=render(iq,None,baseline,tracker,(model,),threshold=-1.)
    logits=load_reader(root,contract,device)(stacked.flatten(0,1))[0]
    if n!=len(iq) or logits.shape!=(len(MODES)*n,10) or not torch.isfinite(logits).all():
        raise ValueError('Real source checkpoint Hankel smoke failed')
    bypass,b=readouts(cal[:2],tracker,model,threshold)
    if b or any(not torch.equal(x,cal[:2]) for x in bypass):
        raise ValueError('Hankel smoke needs untouched legal source clean')
    write(root/'smoke.json',dict(status='PASS',source_U=len(iq),clean_bypassed=len(cal[:2]),
          target_reads=0,dose_max_difference=float((torch.linalg.vector_norm(edits[-2],dim=-1)-
                                                   torch.linalg.vector_norm(edits[-1],dim=-1)).abs().max())))
    print('LORA_CROSS_CHIRP_HANKEL_REAL_CHECKPOINT_PASS',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    actions=cli.add_subparsers(dest='action',required=True)
    pub=actions.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for phase in ('dispatch','smoke','timing','evaluate','score'):
        stage=actions.add_parser(phase)
        stage.add_argument('runtime',type=Path)
        if phase!='dispatch':
            stage.add_argument('row',choices=(*ROUTES,'reserve') if phase=='evaluate' else ('source',))
    args=vars(cli.parse_args())
    action=args.pop('action')
    if action=='publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt=read(args['runtime'])
    if action=='dispatch':
        return dispatch(rt)
    if action=='score':
        return independent_score(rt,modes=MODES,comparisons=PAIRS)
    activate(rt)
    if action=='smoke':
        smoke(rt)
    elif action=='timing':
        timing(rt,reader=readouts,timing_scope='frozen rank and all three Hankel filters; excludes DURA and F')
    elif args['row']=='reserve':
        reserve(rt)
    else:
        evaluate(rt,args['row'])


if __name__=='__main__':
    main()
