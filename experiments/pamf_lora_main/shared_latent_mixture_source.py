"""Compare moment-matched shared LoRa priors on two frozen-recognition routes."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from experiments.cric_dura_comparison.io import read,write
from .action_support_source import independent_score,predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import panel,receive_only,setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader,publish,score
from .data import SOURCE
from .dura_repair import PacketRestorer,complex_iq,real_iq
from .joint_unroll_source import coordinates,source_control
from .packet_runner import schedule
from .private_transport_source import timing as rank_timing
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .shared_latent_mixture import WITHIN_VARIANCE,empirical_centers,replace_common,shared_posteriors
from .spectral_covariance import real_coordinates
from .spectral_covariance_source import RUN as PARENT
from .spectral_unroll import ROUTES
from .variance_bound import spectral_contrast,variance_interval
from .weak_guard_source import parent_model
from .weak_signal_probe import observed_noise_rms


RUN = '20261124-phase1-lora-shared-latent-mixture-source-m2-r01'
PER_RX = 100
MODES = ('direct','reference','rank_half','shared_gaussian','shared_mixture',
         'mixture_delta_half','mixture_delta_full')
PAIRS = (('shared_mixture','shared_gaussian'),('mixture_delta_half','rank_half'),
         ('mixture_delta_full','rank_half'))
ISOLATION = dict(F_reads=0,TX_reads=0,V_reads=0,target_reads=0,trace_reads=0)


def codebook(tracker,training,prior,per_rx):
    balanced=torch.cat([group[:per_rx] for group in training.chunk(len(SOURCE))])
    if len(balanced)!=per_rx*len(SOURCE):
        raise ValueError('LoRa mixture source cohort is incomplete')
    bins=prior.correlated_bins
    measurements=[]
    for batch in balanced.split(32):
        wave=coordinates(tracker,batch)[0]
        measured=torch.fft.fft(wave,norm='ortho').index_select(-1,bins).mean(1)
        measurements.append(real_coordinates(measured))
    diagonal=prior.shared_real-prior.factors.square().sum(-1)
    return empirical_centers(torch.cat(measurements),real_coordinates(prior.mean[bins]),
                             diagonal,prior.factors)


def validate_moments(centers):
    cov=centers.T @ centers / len(centers)
    identity=torch.eye(cov.shape[0],device=cov.device,dtype=cov.dtype)
    mean_error=float(centers.mean(0).abs().amax())
    covariance_error=float((cov+WITHIN_VARIANCE*identity-identity).abs().amax())
    if not mean_error<=1e-5 or not covariance_error<=1e-5:
        raise ValueError('The Gaussian and mixture fit must have equal prior moments')
    return dict(latent_mean_max_error=mean_error,latent_covariance_max_error=covariance_error)


@torch.no_grad()
def source_fit(rt):
    start_time=perf_counter()
    root,device=Path(rt['run_root']),'cuda:0'
    (training,_,_),contract=setup(rt,device)
    parent,_=parent_model(root,contract,device)
    centers=codebook(PacketRestorer('consensus').to(device).eval(),training,parent.clean_detail,PER_RX)
    metrics=validate_moments(centers)
    saved=dict(status='SOURCE_FROZEN',contract=contract,parent=PARENT,centers=centers.cpu(),within=WITHIN_VARIANCE)
    with (root/'source-mixture.pt').open('xb') as file:
        torch.save(saved,file)
    write(root/'source-mixture-complete.json',dict(status='SOURCE_FROZEN',n=len(centers),latent_dim=centers.shape[1],
          state_bytes=centers.numel()*centers.element_size(),fit_seconds=perf_counter()-start_time,**metrics,**ISOLATION))
    print('LORA_SHARED_MIXTURE_FROZEN',len(centers),flush=True)


def load_codebook(root,contract,device):
    saved=torch.load(root/'source-mixture.pt',weights_only=True,map_location=device)
    if (saved['status']!='SOURCE_FROZEN' or saved['contract']!=contract or saved['parent']!=PARENT
            or saved['within']!=WITHIN_VARIANCE or saved['centers'].shape!=(PER_RX*len(SOURCE),4)):
        raise ValueError('Need the source-only moment-matched LoRa mixture from this run')
    return saved['centers']


@torch.no_grad()
def normalized_variants(y,rot,q,model,centers):
    midpoint=torch.lerp(y,model(y,q,rot)[-1],.5)
    spectrum=torch.fft.fft(y,norm='ortho')
    contrast,weights=spectral_contrast(spectrum,q)
    outer=observed_noise_rms(y).square()
    estimates,_=variance_interval(outer,contrast,weights,model.selectors)
    g,m=shared_posteriors(spectrum,estimates @ model.bands,model.clean_detail,centers)
    bins=model.clean_detail.correlated_bins
    raw=spectrum.index_select(-1,bins).mean(1)
    half_spectrum=torch.fft.fft(midpoint,norm='ortho')
    current=half_spectrum.index_select(-1,bins).mean(1)
    delta=m-g
    new=(torch.lerp(raw,g,.5),torch.lerp(raw,m,.5),current+.5*delta,current+delta)
    changes=[replace_common(half_spectrum,bins,value)-half_spectrum for value in new]
    return midpoint,tuple(torch.fft.ifft(change,norm='ortho') for change in changes)


@torch.no_grad()
def readouts(iq,tracker,model,threshold,*,centers):
    y,rot,q,rms=coordinates(tracker,iq)
    active=corruption_score(y)>threshold
    n=int(active.sum())
    if not n:
        return [iq]*(len(MODES)-2),0
    y,rot,q,rms=(x[active] for x in (y,rot,q,rms))
    midpoint,changes=normalized_variants(y,rot,q,model,centers)
    base=iq.clone()
    base[active]+=real_iq((midpoint-y)*rot*rms)
    output=[base]
    for delta in changes:
        edited=base.clone()
        edited[active]+=real_iq(delta*rot*rms)
        output.append(edited)
    return output,n


@torch.no_grad()
def render(iq,_unused,baseline,tracker,models,*,threshold):
    reference,_=baseline(iq)
    output,n=readouts(iq,tracker,models[0],threshold,centers=models[1])
    return torch.stack((iq,reference,*output)),n


def bound_assets(root,contract,device):
    model,threshold=parent_model(root,contract,device)
    return (model,load_codebook(root,contract,device)),threshold


def evaluate(rt,route):
    root,device=Path(rt['run_root']),'cuda:0'
    binding=source_control(rt,route,device)
    models,threshold=bound_assets(root,binding[2],device)
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=threshold),
          binding=binding,frozen_models=models,require_clean_bypass=False,
          replay_fn=receive_only,finalize=freeze,reader_chunk=144)


def reserve(rt):
    def bind(root,contract,device,baseline,tracker):
        models,threshold=bound_assets(root,contract,device)
        return partial(render,_unused=None,baseline=baseline,tracker=tracker,models=models,threshold=threshold)
    predict_clean_reserve(rt,modes=MODES,bind=bind)


@torch.no_grad()
def smoke(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (training,cal,_),contract=setup(rt,device,smoke=True)
    _,_,expected,baseline,tracker=source_control(rt,'full',device)
    if expected!=contract:
        raise ValueError('Source IQ and frozen recognition contract differ')
    model,threshold=parent_model(root,contract,device)
    prototypes=codebook(tracker,training,model.clean_detail,5)
    metrics=validate_moments(prototypes)
    rng=torch.Generator(device=device).manual_seed(2026114631)
    contaminated=real_iq(colored_observation(complex_iq(cal[:3]),rng))
    waves,n=render(contaminated,None,baseline,tracker,(model,prototypes),threshold=-1.)
    scores=load_reader(root,contract,device)(waves.flatten(0,1))[0]
    if n!=len(contaminated) or scores.shape!=(len(MODES)*n,10) or not torch.isfinite(scores).all():
        raise ValueError('Real LoRa codebook and classifier smoke failed')
    clean,n=readouts(cal[:3],tracker,model,threshold,centers=prototypes)
    if n or any(not torch.equal(x,cal[:3]) for x in clean):
        raise ValueError('Frozen clean bypass failed')
    write(root/'smoke.json',dict(status='PASS',source_u=len(contaminated),target_reads=0,**metrics))
    print('LORA_SHARED_LATENT_MIXTURE_REAL_CHECKPOINT_PASS',flush=True)


def timing(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,_,_),contract=setup(rt,device,smoke=True)
    reader=partial(readouts,centers=load_codebook(root,contract,device))
    rank_timing(rt,reader=reader,timing_scope='one frozen rank parent and both shared posteriors; excludes DURA and F')


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    execute=partial(start,module=rt['runner'])
    try:
        for phase,rows in (('smoke',('source',)),('fit',('source',)),('timing',('source',)),
                           ('evaluate',ROUTES),('evaluate',('reserve',))):
            schedule(rt,[(phase,row) for row in rows],execute)
        if execute(rt,'score','source').wait():
            raise RuntimeError('Independent LoRa mixture scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as e:
        write(root/'state.json',dict(status='FAILED',error=str(e)))
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    entries=cli.add_subparsers(dest='action',required=True)
    pub=entries.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for phase in ('dispatch','smoke','fit','timing','evaluate','score'):
        sub=entries.add_parser(phase)
        sub.add_argument('runtime',type=Path)
        if phase!='dispatch':
            sub.add_argument('row',choices=(*ROUTES,'reserve') if phase=='evaluate' else ('source',))
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
    if action in ('smoke','fit','timing'):
        {'smoke':smoke,'fit':source_fit,'timing':timing}[action](rt)
    elif args['row']=='reserve':
        reserve(rt)
    else:
        evaluate(rt,args['row'])


if __name__=='__main__':
    main()
