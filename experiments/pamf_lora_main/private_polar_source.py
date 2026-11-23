"""Source-only frozen-recognition and waveform tests of LoRa private polar edits."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import panel, partition, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates, source_control
from .offband_noise_source import replay
from .packet_runner import schedule
from .private_band_risk_source import BANDS, band_inner, fft, private_projection
from .private_polar import POLAR_MODES, polar_deltas
from .private_transport_bands import named_band_masks
from .private_transport_source import proposed_reads, readouts as parent_reads, timing
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-private-polar-source-m2-r01'
MODES = ('direct','reference','rank_half','wiener_half','wiener_full',*POLAR_MODES)
PAIRS = tuple((name, control) for name in POLAR_MODES for control in ('wiener_half','wiener_full'))
WAVE_MODES = ('wiener_half','wiener_full',*POLAR_MODES)
PER_RX = 30


def variants(y, q, model):
    rank, actions = proposed_reads(y,q,model)
    half, full = actions[0], actions[2]
    return rank, (half, full, *polar_deltas(rank,half,full,named_band_masks(y)))


@torch.no_grad()
def readouts(iq, tracker, model, threshold):
    y, rot, q, rms = coordinates(tracker,iq)
    active = corruption_score(y) > threshold
    n = int(active.sum())
    if not n:
        return [iq]*(len(MODES)-2),0
    y, rot, q, rms = (x[active] for x in (y,rot,q,rms))
    rank, edits = variants(y,q,model)
    midpoint = iq.clone()
    midpoint[active] += real_iq((rank-y)*rot*rms)
    out = [midpoint]
    for delta in edits:
        restored = midpoint.clone()
        restored[active] += real_iq(delta*rot*rms)
        out.append(restored)
    return out,n


@torch.no_grad()
def render(iq,_unused,reference,tracker,models,*,threshold):
    original,_ = reference(iq)
    outputs,n = readouts(iq,tracker,models[0],threshold)
    return torch.stack((iq,original,*outputs)),n


def evaluate(rt,route):
    root,device = Path(rt['run_root']),'cuda:0'
    binding = source_control(rt,route,device)
    model,threshold = parent_model(root,binding[2],device)
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=threshold),
          binding=binding,frozen_models=(model,),require_clean_bypass=False,
          replay_fn=receive_only,finalize=freeze,reader_chunk=144)


def reserve(rt):
    def bind(root,contract,device,baseline,tracker):
        model,threshold = parent_model(root,contract,device)
        return partial(render,_unused=None,reference=baseline,tracker=tracker,models=(model,),threshold=threshold)
    predict_clean_reserve(rt,modes=MODES,bind=bind)


@torch.no_grad()
def smoke(rt):
    root,device = Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract = setup(rt,device,smoke=True)
    _,_,bound,baseline,tracker = source_control(rt,'full',device)
    if bound != contract:
        raise ValueError('Source IQ and frozen LoRa recognition contract differ')
    model,threshold = parent_model(root,contract,device)
    generator = torch.Generator(device=device).manual_seed(2026113627)
    iq = real_iq(colored_observation(complex_iq(cal[:3]),generator))
    checked,n = render(iq,None,baseline,tracker,(model,),threshold=-1.)
    parent,_ = parent_reads(iq,tracker,model,-1.)
    for name,i in (('rank_half',0),('wiener_half',1),('wiener_full',3)):
        if not torch.equal(checked[MODES.index(name)],parent[i]):
            raise ValueError(f'The frozen {name} endpoint changed')
    logits = load_reader(root,contract,device)(checked.flatten(0,1))[0]
    if n != len(iq) or logits.shape != (len(MODES)*len(iq),10) or not torch.isfinite(logits).all():
        raise ValueError('LoRa polar real-checkpoint smoke failed')
    write(root/'smoke.json',dict(status='PASS',source_u=len(iq),threshold=threshold,
          unchanged_parent_rows=['rank_half','wiener_half','wiener_full'],target_reads=0))
    print('LORA_PRIVATE_POLAR_CHECKPOINT_PASS',flush=True)


def wave_quantities(observed, expected, weight, masks):
    error = band_inner(observed-expected,observed-expected,weight,masks).sum(0)
    power = band_inner(expected,expected,weight,masks)
    overlap = torch.complex(band_inner(observed,expected,weight,masks),
                            band_inner(observed,1j*expected,weight,masks))
    result = {}
    for i,label in enumerate(BANDS):
        keep = power[:,i] > 1e-12
        gain = overlap[keep,i] / power[keep,i]
        if not keep.any() or not gain.isfinite().all() or not error[i].isfinite():
            raise ValueError('Private polar waveform statistics are invalid')
        result[label] = dict(n=int(keep.sum()),relative_mse=float(error[i]/power[:,i].sum()),
                             median_real_gain=float(gain.real.median()),
                             median_abs_phase_rad=float(gain.angle().abs().median()))
    return result


@torch.no_grad()
def waveform(rt):
    source = Path(rt['source_root'])
    root = Path(rt['run_root'])
    contract = source_contract(source)
    raw,rx,ids,_ = source_role(source,contract,'U_s')
    _,_,held = partition(rx)
    chosen = np.concatenate([chunk[:PER_RX] for chunk in np.split(held,len(SOURCE))])
    model,threshold = parent_model(root,contract,'cpu')
    tracker = PacketRestorer('consensus').eval()
    rows = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ROUTES:
            rows[route] = {}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs = ((route,view,raw[i],rx[i],ids[i]) for i in chosen)
                iq,noise = (tensor(np.stack(x)) for x in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                restored = {name:[] for name in WAVE_MODES}
                obs, basis, active = [],[],[]
                for batch in iq.split(18):
                    y,rot,q,rms = coordinates(tracker,batch)
                    mask = corruption_score(y) > threshold
                    y,q = y[mask],q[mask]
                    active.append(mask)
                    if not len(y):
                        continue
                    obs.append(y)
                    basis.append((rot*rms)[mask])
                    rank,deltas = variants(y,q,model)
                    for name,delta in zip(WAVE_MODES,deltas,strict=True):
                        restored[name].append(private_projection(fft(rank+delta)))
                used = torch.cat(active)
                if not used.any():
                    raise ValueError('Private polar waveform probe needs active observations')
                observed, matrices = torch.cat(obs), torch.cat(basis)
                # Isolated reference enters only after all readouts have been computed.
                expected = private_projection(fft(observed-complex_iq(noise[used])/matrices))
                masks = named_band_masks(observed).float()
                weights = matrices.abs().square()
                rows[route][view] = dict(n=len(chosen),active=int(used.sum()),by_mode={
                    name:wave_quantities(torch.cat(parts),expected,weights,masks)
                    for name,parts in restored.items()})
                print('LORA_PRIVATE_POLAR_WAVE',route,view,int(used.sum()),flush=True)
    write(root/'source-u-polar-wave.json',dict(status='SOURCE_WAVE_COMPLETE',n=len(chosen),
          target_reads=0,F_reads=0,V_reads=0,TX_reads=0,noise_after_readouts=True,rows=rows))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    execute = partial(start,module=rt['runner'])
    try:
        schedule(rt,[('smoke','source')],execute)
        diagnostics = execute(rt,'waveform','source')
        for phase, rows in (('timing',('source',)),('evaluate',ROUTES),('evaluate',('reserve',))):
            schedule(rt,[(phase,row) for row in rows],execute)
        if execute(rt,'score','source').wait():
            raise RuntimeError('Independent private polar scoring failed')
        if diagnostics.wait():
            raise RuntimeError('Source U private polar waveform probe failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root/'state.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+key,type=Path,required=True)
    for phase in ('dispatch','smoke','waveform','timing','evaluate','score'):
        cmd = actions.add_parser(phase)
        cmd.add_argument('runtime',type=Path)
        if phase != 'dispatch':
            cmd.add_argument('row',choices=(*ROUTES,'reserve') if phase=='evaluate' else ('source',))
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt = read(args['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    if action == 'score':
        return independent_score(rt,modes=MODES,comparisons=PAIRS)
    activate(rt)
    if action == 'smoke':
        smoke(rt)
    elif action == 'waveform':
        waveform(rt)
    elif action == 'timing':
        timing(rt,reader=readouts,timing_scope='one frozen parent with six polar variants; excludes DURA and classifier')
    elif args['row'] == 'reserve':
        reserve(rt)
    else:
        evaluate(rt,args['row'])


if __name__ == '__main__':
    main()
