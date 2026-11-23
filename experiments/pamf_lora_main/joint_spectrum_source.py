"""Train matched LoRa complex-spectrum arms and freeze both routes before scoring."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from experiments.cric_dura_comparison.io import write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import JointSpectrum, objective
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze
from .reference_cancel import generic_observation
from .spectral_block_source import load_models, main
from .variance_transport import match_dose, physical_norm


RUN = '20261124-phase1-lora-joint-spectrum-source-m4-r01'
RECIPES = {name:FitRecipe(name,40_000,partial(JointSpectrum,exchange=name=='exchange'),
                          generic_observation,None,coarse_first=True,batch=24,fit_per_rx=4000,
                          validation_steps=(1800,6000,12000,20000,30000))
           for name in ('independent','exchange')}
READOUTS = ('coarse','full','half')
MODES = ('direct','reference',*(f'{arm}_{mode}' for arm in RECIPES for mode in READOUTS),
         'exchange_matched','own_matched')
COMPARISONS = (('exchange_matched','own_matched'),('exchange_full','exchange_coarse'),
               *((f'exchange_{kind}',f'independent_{kind}') for kind in READOUTS),
               *((mode,base) for mode in MODES[2:] for base in ('reference','direct')))
FIELDS = ('active','outer_common','outer_private','split_coherence','context_level','full_edit_rms_ratio',
          'matched_edit_rms_ratio')
DEVICE = 'cuda:0'


def recover(iq, tracker, models, threshold, *, compare=False, switch='exchange'):
    observed,rotation,quality,rms = coordinates(tracker,iq)
    enabled = corruption_score(observed)>threshold
    count = int(enabled.sum())
    outputs = iq.expand(3*len(models) + (2 if compare else 0),*iq.shape).clone()
    measures = iq.new_zeros((len(iq),len(FIELDS)))
    if not count:
        return outputs,count,measures
    y,phase,q,level = (value[enabled] for value in (observed,rotation,quality,rms))
    proposals = []
    for model in models:
        coarse,fine,estimates = model.stages(y,q,level)
        edit = fine-y
        proposals.extend((coarse-y,edit,.5*edit))
    if compare:
        own = models[-1].stages(y,q,level,**{switch:False})[1]-y
        matched = match_dose(edit,own,level)
        proposals.extend((matched,match_dose(own,edit,level)))
        doses = (physical_norm(torch.stack((edit,matched)),level)/physical_norm(y,level).clamp_min(1e-7))
        measures[enabled] = torch.cat((torch.ones_like(estimates[:,:1]),estimates,doses[...,0,0].mT),dim=1)
    physical = torch.stack(proposals)*phase*level
    outputs[:,enabled] = iq[enabled]+torch.stack((physical.real,physical.imag),dim=-2)
    return outputs,count,measures


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None, switch='exchange'):
    reference,_ = baseline(iq)
    outputs,active,evidence = recover(iq,tracker,models,threshold,compare=True,switch=switch)
    if measurements is not None:
        measurements.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq,reference)),outputs)),active


@torch.no_grad()
def smoke(rt, recipes, modes, *, switch='exchange'):
    root = Path(rt['run_root'])
    (_,source,_),contract = setup(rt,DEVICE,smoke=True)
    models,threshold = load_models(root,contract,DEVICE,recipes)
    _,_,expected,baseline,tracker = source_control(rt,'full',DEVICE)
    if expected!=contract:
        raise ValueError('Expected the existing LoRa source contract')
    reader = load_reader(root,contract,DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(2026112414)
    perturbed = real_iq(generic_observation(complex_iq(source),rng))
    active = perturbed[corruption_score(coordinates(tracker,perturbed)[0]).argmax().view(1)]
    clean = source[corruption_score(coordinates(tracker,source)[0]).argmin().view(1)]
    output,triggered = render(torch.cat((clean,active)),None,baseline,tracker,models,threshold=threshold,switch=switch)
    logits = reader(output.flatten(0,1))[0]
    if triggered!=1 or logits.shape!=(2*len(modes),10) or not torch.isfinite(logits).all():
        raise ValueError('Expected a finite real-checkpoint result with an active restoration')
    if not torch.equal(output[2:,0],clean.expand_as(output[2:,0])):
        raise ValueError('The no-query smoke must preserve every clean sample')
    times = {}
    for name,wave in (('clean',clean),('generic_disturbed',active)):
        delays = []
        for i in range(28):
            torch.cuda.synchronize()
            start = perf_counter()
            recover(wave,tracker,models[-1:],threshold)
            torch.cuda.synchronize()
            if i>=8:
                delays.append(1000*(perf_counter()-start))
        times[name] = float(np.median(delays))
    write(root/'smoke.json',dict(status='PASS',V_reads=0,target_reads=0,timing_b1_ms=times,
          params={name:sum(p.numel() for p in model.parameters()) for name,model in zip(recipes,models,strict=True)},
          device=torch.cuda.get_device_name()))


def evaluate(rt, route, recipes, modes, *, switch='exchange', fields=FIELDS):
    binding = source_control(rt,route,DEVICE)
    models,threshold = load_models(Path(rt['run_root']),binding[2],DEVICE,recipes)
    records = []

    def finalize(root, route, source, chosen, ids, predictions, metrics, modes):
        values = np.concatenate(records).reshape(len(VIEWS),len(chosen),len(fields))
        result = {}
        for view, rows in zip(VIEWS,values,strict=True):
            result[view] = {}
            for group,cells in zip(SOURCE,np.split(rows,len(SOURCE)),strict=True):
                active = cells[cells[:,0]>0]
                medians = dict(zip(fields[1:],np.median(active[:,1:],axis=0),strict=True)) if len(active) else {}
                result[view][group] = dict(active=len(active),**medians)
        write(root/f'{route}-estimates.json',dict(by_view=result))
        freeze(root,route,source,chosen,ids,predictions,metrics,modes)

    score(rt,route,panel=panel,modes=modes,binding=binding,frozen_models=models,replay_fn=receive_only,
          render=partial(render,threshold=threshold,measurements=records,switch=switch),require_clean_bypass=False,
          finalize=finalize)


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,recipes=RECIPES,modes=MODES,comparisons=COMPARISONS,
         calculate=objective,run_smoke=smoke,run_evaluate=evaluate)
