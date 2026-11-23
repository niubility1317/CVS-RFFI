"""Source U audit of whether observed witness frequencies predict LoRa noise."""

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
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_role
from .joint_unroll_source import coordinates
from .native_detail import native_pair
from .offband_noise import conditional_noise, received_profile
from .packet_physics import aligned_view, residual_view
from .refine_runner import start
from .runner import activate


RUN = '20261124-phase1-lora-offband-noise-source-cpu-r01'
SEED = 2026113413
PER_RX = 30
BANDS = ((1,8), (9,32), (33,64))
MARGINS = (0,32)
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0, training_updates=0)


def band(name):
    return f'{name[0]}_to_{name[1]}'


def draw_noise(iq, kind, generator):
    x = complex_iq(iq)
    f = torch.fft.fftfreq(x.shape[-1])
    size = (len(x),1,1)
    position = torch.rand(size,generator=generator) - .5
    periodic = (f-position+.5).remainder(1)-.5
    width = {'white':.12,'broad':.12,'narrow':.017}[kind]
    spectrum = torch.ones_like(periodic) if kind=='white' else .08+(-.5*(periodic/width).square()).exp()
    spectrum = spectrum / spectrum.mean(-1,keepdim=True)
    white = torch.randn(x.shape,dtype=x.dtype,generator=generator)
    variance = (.08+.28*torch.rand((*x.shape[:2],1),generator=generator)).square()
    signal_power = x.abs().square().mean(-1,keepdim=True)
    noise = torch.fft.ifft(torch.fft.fft(white,norm='ortho')*(spectrum*variance*signal_power).sqrt(),norm='ortho')
    return real_iq(x+noise), real_iq(noise), spectrum


def replay(item):
    route,view,iq,rx,identity = item
    if route=='full':
        output,_,noise = native_pair((iq,rx,identity,view,'source-U',None),sidebands=False)
    else:
        output,signal,total = residual_view(iq,view,rx,identity,'source-U',components=True)
        recovered = output[:,0]+1j*output[:,1]
        if not np.allclose(recovered,signal+total,atol=8e-7,rtol=2e-6):
            raise ValueError('Actual residual and isolated noise reference do not match')
        noise = np.stack((total.real,total.imag),axis=1).astype(np.float32)
    expected=aligned_view(iq,view,rx,identity,'source-U',processing_route=route)
    if not np.array_equal(output,expected):
        raise ValueError('Exact isolated replay differs from the registered receiver observation')
    return output,noise


def measurements(value, epsilon, signal, rms, f, supported):
    witnessed=value[...,f]
    real_noise=epsilon[...,f]
    truth=signal[...,f]
    weight=rms.square()
    dims=(1,2)
    power=(real_noise.abs().square()*weight).sum(dims)
    source=(truth.abs().square()*weight).sum(dims)
    err=((real_noise-witnessed).abs().square()*weight).sum(dims)
    edit=(witnessed.abs().square()*weight).sum(dims)
    cross=(witnessed*real_noise.conj()*weight).sum(dims)
    impact=(witnessed*truth.conj()*weight).sum(dims)
    edited_signal_change=-impact.real / source.clamp_min(1e-12)
    valid=(edit>1e-12)&(power>1e-12)
    actual_correlation=cross.real[valid] / (power[valid]*edit[valid]).sqrt()
    total=float(power.sum())
    return dict(n=len(power), rms_edit_median=float(edit.sqrt().median()),
                fraction_under_square_error=float((err<power).float().mean()) if total>1e-12 else None,
                reduction=float((power-err).sum()/power.sum()) if total>1e-12 else None,
                noise_correlation_median=float(actual_correlation.median()) if valid.any() else None,
                signal_real_gain_shift_median=float(edited_signal_change.median()),
                signal_projection_abs_median=float(edited_signal_change.abs().median()),
                density_support=supported)


def evaluate(tracker, iq, noise_iq, isolated_profile=None):
    rows={}
    y,rotation,_,rms=coordinates(tracker,iq)
    spectrum=torch.fft.fft(y,norm='ortho')
    epsilon=torch.fft.fft(complex_iq(noise_iq)*rotation.conj()/rms,norm='ortho')
    signal=spectrum-epsilon
    freq=(spectrum.shape[-1]*torch.fft.fftfreq(spectrum.shape[-1])).round().abs()
    for bounds in BANDS:
        selected=(freq>=bounds[0])&(freq<=bounds[1])
        for offset in MARGINS:
            witness=freq>(bounds[-1]+offset)
            profile=received_profile(spectrum,rotation,witness,share=.5)
            alone=received_profile(spectrum,rotation,witness,share=0.)
            flat=torch.ones_like(profile)
            choices={'received_shared':(spectrum,profile),
                     'received_each':(spectrum,alone), 'flat':(spectrum,flat)}
            if bool(epsilon.abs().amax()>0):
                choices['isolated_noise_witness_same_profile'] = (epsilon,profile)
            if isolated_profile is not None:
                choices['isolated_known_generic_profile'] = (spectrum,isolated_profile)
                choices['isolated_known_profile_and_noise_witness'] = (epsilon,isolated_profile)
            data={}
            for mode,(samples,density) in choices.items():
                proposal=conditional_noise(samples,rotation,density,witness)
                for strength in (0.5,1.):
                    data[f'{mode}_{strength}']=measurements(proposal*strength,epsilon,signal,rms,selected,
                                                           int(witness.sum()))
            rows.setdefault(band(bounds),{})[str(offset)]=data
    return rows


@torch.no_grad()
def smoke(rt):
    root=Path(rt['run_root'])
    (_,cal,_),_=setup(rt,'cpu',smoke=True)
    tracker=PacketRestorer('consensus').eval()
    noisy,noise,density=draw_noise(cal[:2],'narrow',torch.Generator().manual_seed(SEED))
    data=evaluate(tracker,noisy,noise,density)
    expected=data['1_to_8']['0']['flat_1.0']
    visible=data['1_to_8']['0']['received_shared_1.0']
    if expected['rms_edit_median']>1e-6 or not np.isfinite(visible['reduction']):
        raise ValueError('Offband complex covariance failed its received-IQ smoke')
    write(root/'smoke.json',dict(status='PASS',samples=2,flat_prediction_rms=expected['rms_edit_median'],
                                 **ISOLATION))


@torch.no_grad()
def probe(rt):
    started=perf_counter()
    root=Path(rt['run_root'])
    tracker=PacketRestorer('consensus').eval()
    (_,_,held),contract=setup(rt,'cpu')
    clean=torch.cat([chunk[:PER_RX] for chunk in held.chunk(len(SOURCE))])
    report=dict(n=len(clean), role='held_U_s', **ISOLATION)
    generic={}
    for kind in ('white','broad','narrow'):
        noisy,noise,density=draw_noise(clean,kind,torch.Generator().manual_seed(SEED))
        generic[kind]=evaluate(tracker,noisy,noise,density)
        print('OFFBAND_GENERIC_READY',kind,len(clean),flush=True)
    report['generic']=generic
    report['clean']=evaluate(tracker,clean,torch.zeros_like(clean))
    data=Path(rt['source_root'])
    raw,rx,ids,_=source_role(data,contract,'U_s')
    _,_,held_idx=partition(rx)
    selected=np.concatenate([group[:PER_RX] for group in np.split(held_idx,len(SOURCE))])
    report['registered']=registered={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            rows={}
            for view in (VIEWS[3],VIEWS[-1]):
                items=((route,view,raw[i],rx[i],ids[i]) for i in selected)
                y,noise=(tensor(np.stack(group),'cpu')
                         for group in zip(*pool.map(replay,items,chunksize=6),strict=True))
                rows[view]=evaluate(tracker,y,noise)
                print('OFFBAND_REGISTERED_READY',route,view,len(y),flush=True)
            registered[route]=rows
    report['elapsed_seconds']=perf_counter()-started
    write(root/'source-u-offband-results.json',report)


def dispatch(rt):
    folder=Path(rt['run_root'])
    folder.mkdir()
    launch=partial(start,module=rt['runner'])
    try:
        for action in ('smoke','probe'):
            if launch(rt,action,'source').wait():
                raise RuntimeError(f'Source offband {action} failed')
        write(folder/'state.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE'))
    except Exception as e:
        write(folder/'state.json',dict(status='FAILED',error=str(e)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    pub=commands.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for stage in ('dispatch','smoke','probe'):
        entry=commands.add_parser(stage)
        entry.add_argument('runtime',type=Path)
        if stage!='dispatch':
            entry.add_argument('row',choices=('source',))
    args=vars(parser.parse_args())
    action=args.pop('action')
    if action=='publish':
        publish(**args,run=RUN,runner=__spec__.name)
    else:
        rt=read(args['runtime'])
        if action=='dispatch':
            dispatch(rt)
        else:
            activate(rt)
            (smoke if action=='smoke' else probe)(rt)


if __name__=='__main__':
    main()
