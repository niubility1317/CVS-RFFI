"""Attribute a LoRa source accuracy gap to correction direction versus dose."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .data import VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .private_dose import action_parts, band_energy, band_geometry, component_swaps
from .private_secant_source import RUN as PARENT, frozen_models as load_parent
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-private-dose-direction-source-m2-r02'
MODES = ('direct', 'reference', 'paired_half', 'private_gain_half', 'gain_dir_paired_dose',
         'paired_dir_gain_dose', 'private_dir_paired_dose', 'shared_dir_paired_dose')
PAIRS = (('private_gain_half', 'paired_half'),
         ('gain_dir_paired_dose', 'paired_half'), ('private_gain_half', 'paired_dir_gain_dose'),
         ('paired_dir_gain_dose', 'paired_half'), ('private_gain_half','gain_dir_paired_dose'),
         ('private_dir_paired_dose', 'paired_half'), ('shared_dir_paired_dose', 'paired_half'))
BAND_NAMES = ('1_to_8', '9_to_32', '33_to_64', '65_to_128')


def parent_models(root, contract, device):
    predecessor = root.parent / PARENT
    if read(predecessor/'state.json') != {'status': 'SOURCE_SCORED'}:
        raise ValueError('Need both frozen, source-only parent models')
    return load_parent(predecessor, contract, device)


def numerical_evidence(first, second, swaps, rms, bands):
    geometry = torch.stack([band_geometry(a, b, rms, bands)
                            for a,b in zip(action_parts(first,rms), action_parts(second,rms),strict=True)], dim=1)
    errors = []
    for swap, template in zip(swaps,(first,second,first,first),strict=True):
        for measurement,target in zip(action_parts(swap,rms),action_parts(template,rms),strict=True):
            intended = band_energy(target,rms,bands).sqrt()
            actual = band_energy(measurement,rms,bands).sqrt()
            errors.append((actual-intended).abs().amax())
    maximum = torch.stack(errors).max().item()
    if maximum > 2e-6 or not torch.isfinite(geometry).all():
        raise ValueError('LoRa physical band energy is not matched')
    return geometry.cpu(), maximum


@torch.no_grad()
def readouts(iq, tracker, models, threshold, *, telemetry=None):
    y, rotation, q, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    if not active.any():
        if telemetry is not None:
            telemetry.append((torch.empty((0,2,4,4)), 0.))
        return [iq] * (len(MODES)-2), 0
    y, rotation, q, rms = (a[active] for a in (y,rotation,q,rms))
    bands = models[0].bands.clone()
    bands[:,0] = 0
    base_waves = [torch.lerp(y,model(y,q,rotation)[-1],.5)-y for model in models]
    first,second = [torch.fft.fft(delta,norm='ortho') for delta in base_waves]
    swaps = component_swaps(first,second,rms,bands)
    if telemetry is not None:
        telemetry.append(numerical_evidence(first,second,swaps,rms,bands))
    alternatives = (*base_waves,*(torch.fft.ifft(value,norm='ortho') for value in swaps))
    result = []
    for new in alternatives:
        out = iq.clone()
        out[active] += real_iq(new * rotation * rms)
        result.append(out)
    return result,int(active.sum())


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold, telemetry=None):
    reference,_ = baseline(iq)
    restored,n = readouts(iq,tracker,models,threshold,telemetry=telemetry)
    return torch.stack((iq,reference,*restored)),n


def summarize_telemetry(rows):
    geometry = torch.cat([x[0] for x in rows], dim=0)
    # The N607 Torch NumPy bridge and the imported NumPy can produce
    # distinct ndarray types. Numeric lists keep the reduction on one side.
    values = np.asarray(geometry.tolist()).reshape(tuple(geometry.shape))
    report = dict(active=len(values),max_matched_band_rms_error=max(x[1] for x in rows))
    for component,group in zip(('shared','private'),values.transpose(1,0,2,3),strict=True):
        bands = {}
        for key, data in zip(BAND_NAMES,group.transpose(2,0,1),strict=True):
            known = data[:,3] > 0
            paired_rms, gain_rms, cosine, _ = data[known].T
            bands[key] = (dict(supported=int(known.sum()),paired_rms_median=float(np.median(paired_rms)),
                               gain_rms_median=float(np.median(gain_rms)),
                               ratio_median=float(np.median(gain_rms/paired_rms)),
                               direction_cosine_median=float(np.median(cosine)))
                          if known.any() else dict(supported=0))
        report[component] = bands
    return report


@torch.no_grad()
def smoke(rt):
    root,device = Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    _,_,expected,baseline,tracker=source_control(rt,'full',device)
    if contract != expected:
        raise ValueError('LoRa U smoke has a different source contract')
    models,threshold = parent_models(root,contract,device)
    generator = torch.Generator(device=device).manual_seed(2026112403)
    iq = real_iq(colored_observation(complex_iq(cal[:2]),generator))
    evidence=[]
    restored,n = render(iq,None,baseline,tracker,models,threshold=-1.,telemetry=evidence)
    summary = summarize_telemetry([(torch.empty((0,2,4,4)), 0.), *evidence])
    if summary['active'] != len(iq):
        raise ValueError('Mixed bypass and active telemetry smoke failed')
    reader = load_reader(root,contract,device)
    logits = reader(restored.flatten(0,1))[0]
    if n!=len(iq) or logits.shape!=(len(iq)*len(MODES),10) or not torch.isfinite(logits).all():
        raise ValueError('Equal-dose real checkpoint smoke failed')
    write(root/'checkpoint_smoke.json',dict(status='PASS',n_source_u=len(iq),threshold=threshold,
          max_band_rms_error=evidence[0][1],target_reads=0))
    print('LORA_PRIVATE_DOSE_REAL_CHECKPOINT_PASS',flush=True)


@torch.no_grad()
def timing(rt):
    root,device = Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    models,threshold = parent_models(root,contract,device)
    tracker = PacketRestorer('consensus').to(device).eval()
    generator = torch.Generator(device=device).manual_seed(2026112403)
    disturbed = real_iq(colored_observation(complex_iq(cal),generator))
    ix = corruption_score(coordinates(tracker,disturbed)[0]).argmax().view(1)
    report = {}
    for key,iq in (('clean',cal[:1]),('generic',disturbed[ix])):
        elapsed=[]
        for _ in range(32):
            torch.cuda.synchronize()
            start_at = perf_counter()
            _,active = readouts(iq,tracker,models,threshold)
            torch.cuda.synchronize()
            elapsed.append(1000 * (perf_counter()-start_at))
        report[key]=dict(active=active,median_ms=float(np.median(elapsed[8:])))
    write(root/'b1.json',dict(timing=report,n_measurements=24,gpu=torch.cuda.get_device_name(),
          parameters_each=[sum(p.numel() for p in model.parameters()) for model in models],
          scope='combined six readouts with two frozen parents; excludes classifier and mature DURA'))


def evaluate(rt, route):
    root,device = Path(rt['run_root']),'cuda:0'
    binding=source_control(rt,route,device)
    models,threshold=parent_models(root,binding[2],device)
    waves=[]
    telemetry={}
    def finalize(folder,label,source,chosen,ids,predictions,metrics,modes):
        size=len(chosen)
        # score() visits every view in the fixed order, in batches of 18.
        steps=(size+17)//18
        if len(waves)!=steps*len(VIEWS):
            raise ValueError('The edit telemetry does not cover the fixed prediction matrix')
        for i,name in enumerate(VIEWS):
            diag=summarize_telemetry(waves[steps*i:steps*(i+1)])
            if diag['active']!=metrics[name]['active']:
                raise ValueError('The edit telemetry and predictions have different activation counts')
            telemetry[name]=diag
        write(folder/f'{label}-edit-geometry.json',dict(threshold=threshold,views=telemetry))
        freeze(folder,label,source,chosen,ids,predictions,metrics,modes)
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=threshold,telemetry=waves),
          binding=binding,frozen_models=models,require_clean_bypass=False,
          replay_fn=receive_only,finalize=finalize,reader_chunk=144)


def evaluate_reserve(rt):
    def bind(root,contract,device,control,tracker):
        models,threshold = parent_models(root,contract,device)
        return partial(render,_unused=None,baseline=control,tracker=tracker,models=models,threshold=threshold)
    predict_clean_reserve(rt,modes=MODES,bind=bind)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    start_worker = partial(start,module=rt['runner'])
    try:
        for phase,rows in (('smoke',('source',)),('timing',('source',)),
                           ('evaluate',ROUTES),('evaluate',('reserve',))):
            schedule(rt,[(phase,row) for row in rows],start_worker)
        if start_worker(rt,'score','source').wait():
            raise RuntimeError('Independent matched-dose source scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as error:
        write(root/'state.json',dict(status='FAILED',error=str(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    entry=commands.add_parser('publish')
    for option in ('repo','output','context','ssh-config','parent-submit'):
        entry.add_argument('--'+option,type=Path,required=True)
    for action in ('dispatch','smoke','timing','evaluate','score'):
        tool=commands.add_parser(action)
        tool.add_argument('runtime',type=Path)
        if action!='dispatch':
            tool.add_argument('row',choices=(*ROUTES,'reserve') if action=='evaluate' else ('source',))
    opts=vars(parser.parse_args())
    action=opts.pop('action')
    if action=='publish':
        publish(**opts,run=RUN,runner=__spec__.name)
        return
    rt=read(opts['runtime'])
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
        elif opts['row']=='reserve':
            evaluate_reserve(rt)
        else:
            evaluate(rt,opts['row'])


if __name__=='__main__':
    main()
