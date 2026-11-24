"""Separate physical edit strength and direction in the frozen LoRa source panel."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import panel, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, publish, read_frozen, score_frozen
from .noise_floor_diagnostic import ROUTES, dispatch
from .noise_split_diagnostic import replay_trace
from .phase_dose_bursts_source import burst_observation
from .runner import activate
from .spectral_mixture_diagnostic import RUN as PREDECESSOR, MODES as PREVIOUS_MODES, source_model
from .variance_transport import physical_norm


RUN = '20261124-diagnostic-lora-dose-direction-source-m2-r01'
DOSES = (.5, 1., 2., 3., 4.)
SUFFIX = ('half', 'one', 'two', 'three', 'four')
MODES = ('direct', 'reference', *(f'analytic_{s}' for s in SUFFIX),
         *(f'exact_{s}' for s in SUFFIX), 'exact_natural', 'exact_all')
COMPARISONS = (*((f'exact_{s}',f'analytic_{s}') for s in SUFFIX),
               *((f'{p}_{a}',f'{p}_{b}') for p in ('analytic','exact')
                 for a,b in zip(SUFFIX[1:],SUFFIX[:-1])),
               ('exact_all','exact_natural'), ('exact_natural','exact_one'),
               *((name,'reference') for name in MODES[2:]))
FIELDS = ('active', 'analytic_rms_fraction', 'natural_exact_rms_fraction',
          'base_to_natural', 'max_rms_disagreement', 'available_exact')


def edits_by_dose(analytic, exact):
    """The analytic readouts are the same for any exact attachment."""
    dose = physical_norm(analytic, 1.)
    natural = physical_norm(exact, 1.)
    good = natural > 1e-8
    unit = torch.where(good, exact*(dose/natural.clamp_min(1e-8)), analytic)
    factors = dose.new_tensor(DOSES).view(-1,1,1,1)
    return factors*analytic, factors*unit, (dose,natural,good)


def recover(iq, noise, tracker, model, threshold):
    y, phase, _, rms = coordinates(tracker,iq)
    enabled = corruption_score(y) > threshold
    out = iq.expand(len(MODES)-2,*iq.shape).clone()
    details = iq.new_zeros((len(iq),len(FIELDS)))
    frame = phase*rms
    exact = -complex_iq(noise)/frame
    exact = exact - exact.mean(-1,keepdim=True)
    natural = real_iq(exact*frame)
    out[-1] = iq+natural
    count = int(enabled.sum())
    if count:
        y, rms, frame = y[enabled],rms[enabled],frame[enabled]
        base = model.changes(y,rms)[0][1]
        analytic, oracle, (dose,exact_size,good) = edits_by_dose(base*frame,exact[enabled]*frame)
        physical = torch.cat((analytic, oracle))
        ref_size = physical_norm(y,rms).flatten().clamp_min(1e-8)
        physical_a,physical_b = physical.chunk(2)
        delta_size = (physical_norm(physical_a,1.)-physical_norm(physical_b,1.)).abs().amax(0).flatten()
        if not torch.isfinite(physical).all() or (delta_size > 5e-6*(1.+ref_size)).any():
            raise ValueError('Exact and analytic directions must have the same physical IQ RMS')
        out[:-2,enabled] = iq[enabled]+torch.stack((physical.real,physical.imag),dim=-2)
        out[-2,enabled] = out[-1,enabled]
        details[enabled] = torch.stack((torch.ones_like(ref_size),dose.flatten()/ref_size,
            exact_size.flatten()/ref_size,(dose/exact_size.clamp_min(1e-8)).flatten(),delta_size,
            good.flatten().to(ref_size)), -1)
    return out,count,details


def render(iq, attached, baseline, tracker, models, *, threshold, measures=None):
    ref,_ = baseline(iq)
    values, count, evidence = recover(iq, attached, tracker, models[0], threshold)
    if measures is not None:
        measures.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq,ref)),values)),count


@torch.no_grad()
def smoke(rt):
    device = 'cuda:0'
    (_, source, _), contract = setup(rt,device,smoke=True)
    _, model, threshold = source_model(rt,contract)
    _, _, expected, baseline, tracker = source_control(rt,'full',device)
    if expected != contract:
        raise ValueError('LoRa source contract mismatch')
    rng = torch.Generator(device=device).manual_seed(2026112471)
    perturbed = real_iq(burst_observation(complex_iq(source),rng))
    quiet = corruption_score(coordinates(tracker,source)[0]).argmin().view(1)
    active = corruption_score(coordinates(tracker,perturbed)[0]).argmax().view(1)
    iq = torch.cat((source[quiet],perturbed[active]))
    noise = torch.cat((torch.zeros_like(source[quiet]),perturbed[active]-source[active]))
    output,count = render(iq,noise,baseline,tracker,(model,),threshold=threshold)
    logits = load_reader(Path(rt['run_root']),contract,device)(output.flatten(0,1))[0]
    if count!=1 or logits.shape!=(2*len(MODES),10) or not torch.isfinite(logits).all():
        raise ValueError('Invalid real-checkpoint dose diagnostic')
    if not torch.equal(output[2:,0],iq[0].expand_as(output[2:,0])):
        raise ValueError('Expected exact clean bypass')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',diagnostic_only=True,V_reads=0,target_reads=0))


def evaluate(rt,route):
    binding = source_control(rt,route,'cuda:0')
    _, model, threshold = source_model(rt,binding[2])
    predecessor = Path(rt['run_root']).parent / PREDECESSOR
    old_ids, old_predictions, old_metrics = read_frozen(predecessor,route,PREVIOUS_MODES)
    prior_modes = dict(direct='direct',reference='reference',analytic_one='pooled_half',
                       exact_one='exact_equal',exact_all='exact_noncarrier')
    measurements = []

    def finish(root,route,source,chosen,ids,predictions,metrics,modes):
        if ids[chosen].tolist()!=old_ids or metrics!=old_metrics:
            raise ValueError('Expected unchanged source physical IDs and gate')
        for view in VIEWS:
            for name,before in prior_modes.items():
                if not np.array_equal(predictions[view][modes.index(name)],old_predictions[view][PREVIOUS_MODES.index(before)]):
                    raise ValueError(f'Original same-dose reference changed for {view}/{name}')
        samples = np.concatenate(measurements).reshape(len(VIEWS),len(SOURCE),-1,len(FIELDS))
        result={}
        for view,groups in zip(VIEWS,samples,strict=True):
            cells={}
            for receiver,group in zip(SOURCE,groups,strict=True):
                active=group[group[:,0]>0]
                cells[receiver]=dict(active=len(active))
                if len(active):
                    cells[receiver].update(dict(zip(FIELDS[1:],np.median(active[:,1:],0).tolist(),strict=True)))
                    cells[receiver]['available_total']=int(active[:,-1].sum())
            result[view]=cells
        write(root/f'{route}-isolated-estimates.json',dict(diagnostic_only=True,by_view=result))
        freeze(root,route,source,chosen,ids,predictions,metrics,modes)

    score(rt,route,panel=panel,modes=MODES,binding=binding,frozen_models=(model,),replay_fn=replay_trace,
          render=partial(render,threshold=threshold,measures=measurements),require_clean_bypass=True,finalize=finish)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest='action',required=True)
    pub=actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','evaluate','score'):
        worker=actions.add_parser(action)
        worker.add_argument('runtime',type=Path)
        if action!='dispatch':
            worker.add_argument('name',choices=ROUTES if action=='evaluate' else ('source',))
    args=vars(parser.parse_args())
    action=args.pop('action')
    if action=='publish':
        publish(**args,run=RUN,runner=__spec__.name)
        return
    rt=read(args['runtime'])
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
