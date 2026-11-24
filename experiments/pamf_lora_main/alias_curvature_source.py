"""Score a matched-direction ablation using same-packet nonlinear noise probes."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .alias_curvature import alias_probe, nonlinear_change, substitute
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_band_power_source import model_binding
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-alias-curvature-source-m2-r01'
MODES = ('direct', 'reference', 'antithetic_half', 'near_extrapolate', 'near_reverse', 'wide_extrapolate')
PAIRS = (('near_extrapolate', 'near_reverse'), ('near_extrapolate', 'antithetic_half'),
         ('near_reverse', 'antithetic_half'), ('wide_extrapolate', 'near_extrapolate'))


def variants(y, rot, quality, rms, model):
    _, refined = model(y, quality, rot)
    anchor = torch.lerp(y, refined, .5)
    probe = alias_probe(y)
    synthetic = torch.cat((y + probe, y - probe))
    _, augmented = model(synthetic, torch.cat((quality, quality)), torch.cat((rot, rot)))
    above, below = torch.lerp(synthetic, augmented, .5).chunk(2, dim=0)
    change = nonlinear_change(anchor, above, below)
    return anchor, *substitute(anchor, change, rms)


@torch.no_grad()
def readouts(iq, tracker, model, threshold):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    n = int(active.sum())
    if not n:
        return [iq] * (len(MODES) - 2), 0
    observed, rot, quality, rms = (v[active] for v in frame)
    rows = []
    for proposal in variants(observed, rot, quality, rms, model):
        output = iq.clone()
        output[active] += real_iq((proposal - observed) * rot * rms)
        rows.append(output)
    return rows, n


@torch.no_grad()
def render(iq, _, baseline, tracker, model):
    reference, _ = baseline(iq)
    out, active = readouts(iq, tracker, *model)
    return torch.stack((iq, reference, *out)), active


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    score(rt, route, panel=panel, modes=MODES, render=render, binding=binding,
          frozen_models=model_binding(rt,binding[2],device=device), replay_fn=receive_only,
          finalize=freeze, reader_chunk=144, require_clean_bypass=False)


def reserve(rt):
    def binding(root, contract, device, baseline, tracker):
        model = model_binding(dict(run_root=str(root)),contract,device=device)
        return lambda iq: render(iq,None,baseline,tracker,model)
    predict_clean_reserve(rt,modes=MODES,bind=binding)


@torch.no_grad()
def smoke(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    _,_,matched,baseline,tracker=source_control(rt,'full',device)
    if matched!=contract:
        raise ValueError('Smoke source contract mismatch')
    model,threshold=model_binding(rt,contract,device=device)
    rng=torch.Generator(device=device).manual_seed(2026115531)
    iq=real_iq(colored_observation(complex_iq(cal),rng))
    touched=corruption_score(coordinates(tracker,iq)[0])>threshold
    tested=torch.cat((cal[:2],iq[touched][:2]))
    waves,n=render(tested,None,baseline,tracker,(model,threshold))
    f=load_reader(root,contract,device)
    logits=f(waves.flatten(0,1))[0]
    if (n!=2 or logits.shape!=(len(tested)*len(MODES),10) or not logits.isfinite().all()
            or not torch.equal(waves[2:,:2],cal[:2].expand_as(waves[2:,:2]))
            or torch.equal(waves[3,2:],waves[4,2:])):
        raise ValueError('Real source checkpoint alias probe or bypass failed')
    size=torch.linalg.vector_norm((waves[3:5,2:]-waves[2,2:]).flatten(2),dim=-1)
    torch.testing.assert_close(size[0],size[1],atol=1e-5,rtol=2e-5)
    write(root/'smoke.json',dict(status='PASS',source_U=len(tested),active=n,target_reads=0,
          dose_max_difference=float((size[0]-size[1]).abs().max())))
    print('LORA_ALIAS_CURVATURE_REAL_CHECKPOINT_PASS',flush=True)


@torch.no_grad()
def timing(rt):
    device='cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    model=model_binding(rt,contract,device=device)
    tracker=PacketRestorer('consensus').to(device).eval()
    rng=torch.Generator(device=device).manual_seed(2026115531)
    iq=real_iq(colored_observation(complex_iq(cal),rng))
    strongest=corruption_score(coordinates(tracker,iq)[0]).argmax().view(1)
    periods={}
    for label,data in (('clean',cal[:1]),('generic',iq[strongest])):
        elapsed=[]
        for _ in range(32):
            torch.cuda.synchronize()
            begun=perf_counter()
            readouts(data,tracker,*model)
            torch.cuda.synchronize()
            elapsed.append(1000*(perf_counter()-begun))
        periods[label]=float(np.median(elapsed[8:]))
    write(Path(rt['run_root'])/'b1.json',dict(gpu=torch.cuda.get_device_name(),median_ms=periods,
          n_measurements=24,restoration_forwards_when_active=3,
          scope='one anchor and a batch of two probes, excludes DURA and F'))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    run=partial(start,module=rt['runner'])
    try:
        for action,names in (('smoke',('source',)),('timing',('source',)),
                             ('evaluate',ROUTES),('evaluate',('reserve',))):
            schedule(rt,[(action,name) for name in names],run)
        if run(rt,'score','source').wait():
            raise RuntimeError('Independent LoRa source scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root/'state.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    commands=cli.add_subparsers(dest='action',required=True)
    pub=commands.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for phase in ('dispatch','smoke','timing','evaluate','score'):
        runner=commands.add_parser(phase)
        runner.add_argument('runtime',type=Path)
        if phase!='dispatch':
            runner.add_argument('row',choices=(*ROUTES,'reserve') if phase=='evaluate' else ('source',))
    args=vars(cli.parse_args())
    phase=args.pop('action')
    if phase=='publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt=read(args['runtime'])
    if phase=='dispatch':
        return dispatch(rt)
    if phase=='score':
        return independent_score(rt,modes=MODES,comparisons=PAIRS)
    activate(rt)
    if phase=='smoke':
        return smoke(rt)
    if phase=='timing':
        return timing(rt)
    return reserve(rt) if args['row']=='reserve' else evaluate(rt,args['row'])


if __name__=='__main__':
    main()
