"""Isolate components of the same frozen LoRa learned correction from received IQ."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from experiments.cric_dura_comparison.io import read, write

from . import joint_spectrum_source as frozen
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .complex_vector_source import load_reader, score as predict
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, publish, read_frozen, score_frozen
from .noise_floor_diagnostic import ROUTES, dispatch
from .reference_cancel import generic_observation
from .runner import activate
from .spectral_block_source import load_models
from .variance_transport import match_dose, physical_norm


RUN = '20261124-diagnostic-lora-learned-split-source-m2-r01'
CONTEXT = ('direct','reference','exchange_full','exchange_half')
MODES = (*CONTEXT,'common_only','private_only','common_with_magnitude','common_with_phase',
         'common_magnitude_matched','common_phase_matched','common_half')
COMPARISONS = (('exchange_full','common_only'),('exchange_full','private_only'),
               ('common_with_magnitude','common_only'),('common_with_phase','common_only'),
               ('common_phase_matched','common_magnitude_matched'),
               *((mode,base) for mode in MODES[4:] for base in ('direct','reference','exchange_full')))
FIELDS = ('common','private','magnitude','phase','matched_private')
DEVICE = 'cuda:0'


def component_edits(raw, fine, scale):
    edit = fine-raw
    public = edit.mean(1,keepdim=True).expand_as(edit)
    private = edit-public
    original_private = raw-raw.mean(1,keepdim=True)
    observed = torch.fft.fft(original_private,norm='ortho')
    restored = observed + torch.fft.fft(private,norm='ortho')
    use = torch.arange(observed.shape[-1],device=observed.device)>0

    def hybrid(absolute, angle):
        wanted = torch.polar(absolute.abs(),angle.angle())
        move = (wanted-observed)*use
        centered = move-move.mean(1,keepdim=True)
        return torch.fft.ifft(centered,norm='ortho')

    magnitude = hybrid(restored,observed)
    phase = hybrid(observed,restored)
    matched_magnitude = match_dose(magnitude,phase,scale)
    matched_phase = match_dose(phase,magnitude,scale)
    edits = torch.stack((edit,.5*edit,public,private,public+magnitude,public+phase,
                         public+matched_magnitude,public+matched_phase,.5*public))
    measure = physical_norm(torch.stack((public,private,magnitude,phase,matched_phase)),scale)[...,0,0].mT
    return edits,measure


def original(rt, contract):
    parent = Path(rt['run_root']).parent/frozen.RUN
    if read(parent/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Expected the complete dual-route frozen parent')
    models,threshold = load_models(parent,contract,DEVICE,frozen.RECIPES)
    return parent,models[-1],threshold


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None):
    base,_ = baseline(iq)
    y,rotation,q,scale = coordinates(tracker,iq)
    active = corruption_score(y)>threshold
    result = iq.expand(len(MODES)-2,*iq.shape).clone()
    evidence = iq.new_zeros((len(iq),len(FIELDS)+1))
    if active.any():
        observed,quality,level = y[active],q[active],scale[active]
        _,fine,_ = models[0].stages(observed,quality,level)
        edits,measure = component_edits(observed,fine,level)
        physical = edits * rotation[active] * level
        result[:,active] = iq[active] + torch.stack((physical.real,physical.imag),dim=-2)
        denom = physical_norm(observed,level)[:,0,0].clamp_min(1e-8)
        evidence[active] = torch.cat((torch.ones_like(denom[:,None]),measure/denom[:,None]),dim=1)
    if measurements is not None:
        measurements.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq,base)),result)),int(active.sum())


@torch.no_grad()
def smoke(rt):
    (_,source,_),contract = setup(rt,DEVICE,smoke=True)
    _,model,threshold = original(rt,contract)
    _,_,bound,baseline,tracker = source_control(rt,'full',DEVICE)
    if contract!=bound:
        raise ValueError('The frozen recovery must match the legal source contract')
    reader = load_reader(Path(rt['run_root']),contract,DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(2026112414)
    changed = real_iq(generic_observation(complex_iq(source),rng))
    good = source[corruption_score(coordinates(tracker,source)[0]).argmin().view(1)]
    bad = changed[corruption_score(coordinates(tracker,changed)[0]).argmax().view(1)]
    input_iq = torch.cat((good,bad))
    options,active = render(input_iq,None,baseline,tracker,(model,),threshold=threshold)
    output = reader(options.flatten(0,1))[0]
    if active!=1 or not torch.equal(options[2:,0],good[0].expand_as(options[2:,0])):
        raise ValueError('Candidate readouts must preserve the original guard')
    if output.shape!=(2*len(MODES),10) or not torch.isfinite(output).all():
        raise ValueError('Expected finite real frozen-reader outputs')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',target_reads=0,V_reads=0,diagnostic_only=True))


def evaluate(rt, route):
    binding = source_control(rt,route,DEVICE)
    parent,model,threshold = original(rt,binding[2])
    previous_ids,previous,marks = read_frozen(parent,route,frozen.MODES)
    indices = [frozen.MODES.index(mode) for mode in CONTEXT]
    records = []

    def finish(root, route, source, chosen, ids, predictions, metrics, modes):
        if ids[chosen].tolist()!=previous_ids or metrics!=marks:
            raise ValueError('The diagnostic changed the frozen physical source panel or guard')
        if any(not np.array_equal(predictions[view][:len(CONTEXT)],previous[view][indices]) for view in VIEWS):
            raise ValueError('The original frozen diagnostic decisions differ')
        array = np.concatenate(records).reshape(len(VIEWS),len(chosen),len(FIELDS)+1)
        groups = {}
        for view, group in zip(VIEWS,array,strict=True):
            groups[view] = {}
            for key, rows in zip(SOURCE,np.split(group,len(SOURCE)),strict=True):
                active = rows[rows[:,0]>0.]
                groups[view][key] = dict(active=len(active),median=np.median(active[:,1:],axis=0).tolist() if len(active) else [])
        write(root/f'{route}-edits.json',dict(fields=FIELDS,by_view=groups,diagnostic_only=True))
        freeze(root,route,source,chosen,ids,predictions,metrics,modes)

    predict(rt,route,panel=panel,modes=MODES,binding=binding,frozen_models=(model,),replay_fn=receive_only,
            render=partial(render,threshold=threshold,measurements=records),require_clean_bypass=True,finalize=finish)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','evaluate','score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime',type=Path)
        if action!='dispatch':
            worker.add_argument('name',choices=ROUTES if action=='evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args,run=RUN,runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        score_frozen(rt,MODES,comparisons=COMPARISONS)
    else:
        activate(rt)
        if action=='smoke':
            smoke(rt)
        else:
            evaluate(rt,args['name'])


if __name__=='__main__':
    main()
