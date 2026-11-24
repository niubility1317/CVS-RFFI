"""Audit how receiver coordinate noise enters private LoRa chirp spectra."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .channel_blind_source import partition, setup
from .complex_vector_source import publish
from .data import SOURCE,VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .frame_leakage import SUBSETS, frame_factors, private_terms, received_frames
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates
from .offband_noise_source import replay
from .private_band_risk_source import BANDS, fft, private_projection
from .private_transport_bands import named_band_masks
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-frame-leakage-source-cpu-r01'
SEED = 2026114399
PER_RX = 30
ISOLATION = dict(F_reads=0,TX_reads=0,V_reads=0,target_reads=0,training_updates=0)


def spectrum_power(spec, weight, masks):
    return (spec.abs().square()*weight).sum(1) @ masks.T


def normalized_band_summary(error, leak, shifted, expected, noise, weight, masks, selection):
    signals = tuple(spectrum_power(x[selection],weight[selection],masks)
                    for x in (error,leak,shifted,expected,noise))
    rows = {}
    for i,b in enumerate(BANDS):
        e,l,h,p,n = (x[:,i] for x in signals)
        valid = p > 1e-12
        if not valid.all():
            raise ValueError('Expected nonzero noise-free reference in every LoRa private band')
        exact = p.sum()
        rows[b] = dict(frame_error_to_private=float(e.sum()/exact),
                       leaked_common_to_private=float(l.sum()/exact),
                       private_deformation_to_private=float(h.sum()/exact),
                       cross_term_to_private=float((e-l-h).sum()/exact),
                       added_noise_to_private=float(n.sum()/exact),
                       median_frame_error_to_private=float((e/p).median()),
                       p90_frame_error_to_private=float(torch.quantile(e/p,.9)))
    return rows


def evaluate_frames(tracker, capture, isolated_signal, groups):
    measured = received_frames(tracker,capture)
    # The candidate coordinates freeze before the second argument reaches the estimator.
    signal = complex_iq(isolated_signal)
    received = complex_iq(capture)
    _,ref_rotation,_,ref_rms = coordinates(tracker,isolated_signal)
    oracle = ref_rotation,ref_rms
    normalized = signal*ref_rotation.conj()/ref_rms
    noise = received-signal
    weights = ref_rms.square()
    masks = named_band_masks(normalized).float()
    pooled = dict(pooled=torch.ones(len(normalized),dtype=torch.bool))
    pooled.update({key:tensor(groups==key) for key in SOURCE if bool((groups==key).any())})
    out = {}

    def measure(name, factors, include, noise_input, *, coordinates_record):
        phi,slope,gain = factors
        mult = (phi if include[0] else 1.)*(slope if include[1] else 1.)*(gain if include[2] else 1.)
        expected,leak,shifted = private_terms(normalized,mult)
        specs = tuple(fft(x) for x in (leak+shifted,leak,shifted,expected,noise_input))
        bands = {group:normalized_band_summary(*specs,weights,masks,pick) for group,pick in pooled.items()}
        out[name] = dict(bands=bands)
        if coordinates_record:
            steps = (slope[...,1:]*slope[...,:-1].conj()).mean(-1).angle()*(slope.shape[-1]-1)/2
            out[name]['coordinate_errors'] = dict(
                center_phase_p90_rad=float(torch.quantile(phi.angle().abs().flatten(),.9)),
                slope_p90_rad_per_half_chirp=float(torch.quantile(steps.abs().flatten(),.9)),
                rms_log_ratio_p90=float(torch.quantile(gain.log().abs().flatten(),.9)))

    base = frame_factors(measured['tracked'],oracle)
    for label in SUBSETS:
        bits = tuple(p=='1' for p in label)
        rotation_factor = (base[0] if bits[0] else 1.)*(base[1] if bits[1] else 1.)
        scale_factor = base[2] if bits[2] else 1.
        add = private_projection(noise*ref_rotation.conj()/ref_rms*rotation_factor*scale_factor)
        measure(label,base,bits,add,coordinates_record=False)
    for label, frame in measured.items():
        factors = frame_factors(frame,oracle)
        r,a = frame
        add = private_projection(noise*r.conj()/a)
        measure(label,factors,(True,True,True),add,coordinates_record=True)
    return out


@torch.no_grad()
def smoke(rt):
    (_,cal,_),_ = setup(rt,'cpu',smoke=True)
    tracker = PacketRestorer('consensus').eval()
    capture = real_iq(colored_observation(complex_iq(cal[:3]),torch.Generator().manual_seed(SEED)))
    test = evaluate_frames(tracker,capture,cal[:3],np.repeat(SOURCE[0],len(capture)))
    for b in BANDS:
        ideal = test['000']['bands']['pooled'][b]
        actual = test['111']['bands']['pooled'][b]
        tracked = test['tracked']['bands']['pooled'][b]
        if ideal['frame_error_to_private']!=0 or actual['frame_error_to_private']<=0:
            raise ValueError('Invalid source no-query coordinate attribution smoke')
        torch.testing.assert_close(actual['frame_error_to_private'],tracked['frame_error_to_private'])
        torch.testing.assert_close(actual['added_noise_to_private'],tracked['added_noise_to_private'],rtol=1e-5,atol=1e-7)
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',source_U=len(capture),**ISOLATION))
    print('LORA_FRAME_LEAKAGE_SOURCE_SMOKE_PASS',flush=True)


@torch.no_grad()
def probe(rt):
    start_time = perf_counter()
    root,source = Path(rt['run_root']),Path(rt['source_root'])
    contract = source_contract(source)
    (_,_,held),bound = setup(rt,'cpu')
    if bound != contract:
        raise ValueError('LoRa source replay and clean data contracts differ')
    raw,rx,ids,_ = source_role(source,contract,'U_s')
    _,_,held_ids = partition(rx)
    chosen=np.concatenate([chunk[:PER_RX] for chunk in np.split(held_ids,len(SOURCE))])
    samples = torch.cat([chunk[:PER_RX] for chunk in held.chunk(len(SOURCE))])
    groups=rx[chosen]
    tracker = PacketRestorer('consensus').eval()
    noisy=real_iq(colored_observation(complex_iq(samples),torch.Generator().manual_seed(SEED)))
    results = dict(clean=evaluate_frames(tracker,samples,samples,groups),
                   generic=evaluate_frames(tracker,noisy,samples,groups))
    print('LORA_FRAME_LEAKAGE_GENERIC',len(chosen),flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ROUTES:
            results[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs=((route,view,raw[i],rx[i],ids[i]) for i in chosen)
                iq,noise=(tensor(np.stack(x)) for x in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                reference = iq-noise
                results[route][view] = evaluate_frames(tracker,iq,reference,groups)
                print('LORA_FRAME_LEAKAGE_READY',route,view,len(chosen),flush=True)
    write(root/'source-u-frame-leakage.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',held_U=len(chosen),
          role='held_U_s',runtime_seconds=perf_counter()-start_time,
          isolated_reference_only_after_coordinate_estimation=True,
          factor_order=['center_phase','slope','rms'],results=results,**ISOLATION))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    try:
        for phase in ('smoke','probe'):
            if start(rt,phase,'source',module=rt['runner']).wait():
                raise RuntimeError(f'LoRa frame leakage {phase} failed')
        write(root/'state.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE'))
    except Exception as e:
        write(root/'state.json',dict(status='FAILED',error=str(e)))
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    commands=cli.add_subparsers(dest='action',required=True)
    pub=commands.add_parser('publish')
    for key in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+key,type=Path,required=True)
    for phase in ('dispatch','smoke','probe'):
        sub=commands.add_parser(phase)
        sub.add_argument('runtime',type=Path)
        if phase!='dispatch':
            sub.add_argument('row',choices=('source',))
    args=vars(cli.parse_args())
    action=args.pop('action')
    if action=='publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt=read(args['runtime'])
    if action=='dispatch':
        return dispatch(rt)
    activate(rt)
    (smoke if action=='smoke' else probe)(rt)


if __name__=='__main__':
    main()
