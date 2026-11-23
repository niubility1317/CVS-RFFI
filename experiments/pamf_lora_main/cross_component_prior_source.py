"""Cross-receiver source-only evidence for LoRa shared-to-private spectral prediction."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .channel_blind_source import partition, setup
from .complex_vector_source import publish
from .cross_component_prior import (component_features, context_center, contextual_private,
                                    fit_regression, private_spectrum)
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .factorized_chirps import FactorizedChirps
from .full_packet_views import source_role
from .joint_unroll_source import coordinates
from .offband_noise_source import replay
from .refine_runner import start
from .runner import activate
from .spectral_covariance import SharedSpectralPrior, real_coordinates
from .variance_bound import spectral_contrast, variance_interval
from .weak_signal_probe import observed_noise_rms


RUN = '20261124-phase1-lora-cross-component-prior-source-cpu-r02'
N_FIT_RX = 200
N_REPLAY_RX = 30
SEED = 2026113551
BANDS = ('1_to_8','9_to_32')
INPUTS = ('public','public_order')
SCOPE = dict(F_reads=0, V_reads=0, TX_reads=0, target_reads=0, fitted_on_practical=False)


def fourier(wave):
    return torch.fft.fft(wave,norm='ortho')


def masks():
    position=(256*torch.fft.fftfreq(256)).round().abs()
    return ((position>0)&(position<=64),(position>0)&(position<=32),
            ((position>0)&(position<=8),(position>8)&(position<=32)))


def frame(tracker, waves):
    return tuple(torch.cat(parts) for parts in
                 zip(*(coordinates(tracker,batch) for batch in waves.split(24)),strict=True))


def train(clean):
    shared,near,_=masks()
    features=component_features(clean,shared,near)
    target=private_spectrum(clean[...,near])
    # Store a matched diagonal residual variance for all context arms.
    fixed=target.mean(0)
    variance=(target-fixed).abs().square().mean(0).clamp_min(1e-9)
    fits={key:fit_regression(x,real_coordinates(target)) for key,x in zip(INPUTS,features,strict=True)}
    return fixed,variance,fits


def centers(spectrum, learned, *, order_spectrum=None):
    shared,near,_=masks()
    fixed,_,fits=learned
    features=component_features(spectrum,shared,near,order_spectrum=order_spectrum)
    return {'mean':fixed[None].expand(len(spectrum),-1,-1),
            **{key:context_center(fits[key],x) for key,x in zip(INPUTS,features,strict=True)}}


def fidelity(estimate,target):
    power=target.abs().square().sum()
    squared=(estimate-target).abs().square().sum()
    cross=estimate*target.conj()
    reference=target.abs().square().sum((1,2))
    valid=reference>1e-12
    amp=cross.sum((1,2))[valid]/reference[valid]
    return dict(n=len(target),energy=float(power),squared_error=float(squared),
                relative_error=float(squared/power.clamp_min(1e-12)),
                median_gain=float(amp.real.median()) if valid.any() else 0.,
                median_absolute_phase=float(amp.angle().abs().median()) if valid.any() else 0.)


def clean_result(spectrum, learned):
    _,near,parts=masks()
    outputs=centers(spectrum,learned)
    target=private_spectrum(spectrum[...,near])
    summary={}
    for key,part in zip(BANDS,parts,strict=True):
        table={}
        for name,value in outputs.items():
            table[name]=fidelity(value[...,part[near]],target[...,part[near]])
        summary[key]=table
    return summary


def condition_on_observation(measured, spec, q, learned, prior, geometry):
    _,near,_=masks()
    moment,weight=spectral_contrast(spec,q)
    band_noise,_=variance_interval(observed_noise_rms(measured).square(),moment,weight,geometry.selectors)
    noise=band_noise @ geometry.bands
    ordinary=prior.mean + prior.innovation(spec-prior.mean,noise,True)
    valid_variance=learned[1]
    c=measured.shape[1]
    projected=(1-2/c)*noise + noise.sum(1,keepdim=True)/c**2
    gain=valid_variance/(valid_variance+projected[...,near].clamp_min(1e-8))
    denoised=centers(ordinary,learned,order_spectrum=spec)
    raw=centers(spec,learned)
    candidates={'direct':spec,'analytic_joint':ordinary}
    for name,center in denoised.items():
        candidates[name]=contextual_private(spec,ordinary,near,center,gain)
    for name in INPUTS:
        candidates[f'{name}_raw_context']=contextual_private(spec,ordinary,near,raw[name],gain)
    return candidates


def noisy_result(measured,rotation,rms,q,reference, learned, prior, geometry):
    spec=fourier(measured)
    candidates=condition_on_observation(measured,spec,q,learned,prior,geometry)
    wanted=fourier(reference*rotation.conj()/rms)
    _,_,parts=masks()
    results={}
    for key,part in zip(BANDS,parts,strict=True):
        true=private_spectrum(wanted[...,part])
        entries={}
        for name,choice in candidates.items():
            entries[name]=fidelity(private_spectrum(choice[...,part]),true)
            half=torch.lerp(spec[...,part],choice[...,part],.5)
            entries[name+'_half']=fidelity(private_spectrum(half),true)
        results[key]=entries
    return results


def independent_white(x,generator):
    wave=complex_iq(x)
    noise=torch.randn(wave.shape,dtype=wave.dtype,generator=generator)
    scale=(.04+.22*torch.rand((*wave.shape[:2],1),generator=generator))
    strength=wave.abs().square().mean(-1,keepdim=True).sqrt()
    return real_iq(wave + strength*scale*noise)


@torch.no_grad()
def smoke(rt):
    root=Path(rt['run_root'])
    (fit,_,held),_=setup(rt,'cpu',smoke=True)
    tracker=PacketRestorer('consensus').eval()
    clean=fourier(frame(tracker,fit)[0])
    learned=train(clean)
    result=clean_result(fourier(frame(tracker,held)[0]),learned)['1_to_8']['public_order']
    if not np.isfinite(result['relative_error']):
        raise ValueError('Receiver clean source regression produced an invalid estimate')
    write(root/'smoke.json',dict(status='PASS',fit_packets=len(fit),hold_packets=len(held),**SCOPE))


@torch.no_grad()
def probe(rt):
    start_time=perf_counter()
    out=Path(rt['run_root'])
    (training,_,held),contract=setup(rt,'cpu')
    tracker=PacketRestorer('consensus').eval()
    geometry=FactorizedChirps((8,8,8,8),coupled=True).eval()
    bank=[fourier(frame(tracker,block[:N_FIT_RX])[0]) for block in training.chunk(len(SOURCE))]
    fixed={}
    generic={}
    fitted={}
    for index,(rx,clean) in enumerate(zip(SOURCE,held.chunk(len(SOURCE)),strict=True)):
        fit=torch.cat([numeric for k,numeric in enumerate(bank) if k!=index])
        learned=train(fit)
        prior=SharedSpectralPrior()
        prior.fit(fit)
        fitted[rx]=(learned,prior)
        clean_spectrum=fourier(frame(tracker,clean)[0])
        fixed[rx]=clean_result(clean_spectrum,learned)
        noise=independent_white(clean,torch.Generator().manual_seed(SEED+index))
        y,rot,q,rms=frame(tracker,noise)
        generic[rx]=noisy_result(y,rot,rms,q,complex_iq(clean),learned,prior,geometry)
        print('CROSS_COMPONENT_GENERIC_READY',rx,len(clean),flush=True)
    source=Path(rt['source_root'])
    raw,rxs,ids,_=source_role(source,contract,'U_s')
    _,_,hold_idx=partition(rxs)
    take=np.concatenate([block[:N_REPLAY_RX] for block in np.split(hold_idx,len(SOURCE))])
    registered={}
    with ProcessPoolExecutor(max_workers=6) as processes:
        for route in ('full','residual'):
            by_view={}
            for view in (VIEWS[3],VIEWS[-1]):
                items=((route,view,raw[i],rxs[i],ids[i]) for i in take)
                capture,noise=(tensor(np.stack(v),'cpu')
                               for v in zip(*processes.map(replay,items,chunksize=6),strict=True))
                x,rot,q,rms=frame(tracker,capture)
                reference=complex_iq(capture)-complex_iq(noise)
                folds={}
                for idx,rx in enumerate(SOURCE):
                    span=slice(idx*N_REPLAY_RX,(idx+1)*N_REPLAY_RX)
                    encoder,native=fitted[rx]
                    folds[rx]=noisy_result(x[span],rot[span],rms[span],q[span],reference[span],
                                           encoder,native,geometry)
                by_view[view]=folds
                print('CROSS_COMPONENT_REGISTERED_READY',route,view,len(capture),flush=True)
            registered[route]=by_view
    sizes={k:b.bytes() for k,b in fitted[SOURCE[0]][0][2].items()}
    write(out/'source-u-cross-component.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',
          n_clean=len(held),n_generic=len(held),n_registered=len(take),fit_per_fold=(len(SOURCE)-1)*N_FIT_RX,
          regression_bytes=sizes,seconds=perf_counter()-start_time,clean=fixed,generic=generic,
          registered=registered,**SCOPE))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    launch=partial(start,module=rt['runner'])
    try:
        for action in ('smoke','probe'):
            if launch(rt,action,'source').wait():
                raise RuntimeError(f'Source cross-component {action} failed')
        write(root/'state.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE'))
    except Exception as e:
        write(root/'state.json',dict(status='FAILED',error=str(e)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    pub=commands.add_parser('publish')
    for key in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','probe'):
        entry=commands.add_parser(action)
        entry.add_argument('runtime',type=Path)
        if action!='dispatch':
            entry.add_argument('row',choices=('source',))
    options=vars(parser.parse_args())
    action=options.pop('action')
    if action=='publish':
        publish(**options,run=RUN,runner=__spec__.name)
    else:
        rt=read(options['runtime'])
        if action=='dispatch':
            dispatch(rt)
        else:
            activate(rt)
            (smoke if action=='smoke' else probe)(rt)


if __name__=='__main__':
    main()
