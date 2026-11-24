"""Train matched LoRa restorers with and without guarded slope refinement."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import independent_score, predict_clean_reserve
from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .crossfit_pilot import paired_action_inputs
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .factorized_chirps_source import calibrate
from .gated_slope import THRESHOLD, guarded_coordinates, selected_pooled_frame
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective
from .packet_runner import schedule
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_covariance import SharedSpectrumRecovery
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-gated-slope-train-source-m4-r01'
RECIPES = {
    name: FitRecipe(name,STEPS,partial(SharedSpectrumRecovery,WIDTHS,correlated=True),
                    colored_observation,SharedSpectrumRecovery.forward,initialize=calibrate)
    for name in ('matched','pooled')
}
MODES = ('direct','reference','matched_half','pooled_half','inference_only_half','matched_full','pooled_full')
PAIRS = (('pooled_half','matched_half'), ('inference_only_half','matched_half'),
         ('pooled_half','inference_only_half'), ('pooled_full','matched_full'))


def restore(iq, active, model, frame, amounts):
    y, rot, q, rms = frame
    _, cleaned = model(y,q,rot)
    full = real_iq((cleaned - y)*rot*rms)
    outputs = []
    for amount in amounts:
        value = iq.clone()
        value[active] += amount*full
        outputs.append(value)
    return outputs


@torch.no_grad()
def render(iq,_unused,reference,tracker,models):
    control,_ = reference(iq)
    frame = coordinates(tracker,iq)
    active = corruption_score(frame[0]) > THRESHOLD
    n = int(active.sum())
    if not n:
        return torch.stack((iq,control,*([iq]*(len(MODES)-2)))),0
    frame = tuple(x[active] for x in frame)
    updated = selected_pooled_frame(tracker,iq[active],frame)
    a,b = restore(iq,active,models[0],frame,(.5,1.))
    c,d = restore(iq,active,models[1],updated,(.5,1.))
    e, = restore(iq,active,models[0],updated,(.5,))
    return torch.stack((iq,control,a,c,e,b,d)),n


def load_arms(root,contract,device):
    checkpoints = [load_model(root,recipe,contract,device) for recipe in RECIPES.values()]
    # Keep the published threshold fixed for both routes. The independent
    # source-U clean recomputation checks the calibration lineage.
    if not all(abs(value-THRESHOLD)<1e-6 for _,value in checkpoints):
        raise ValueError('Expected the original shared-covariance source U clean calibration')
    return tuple(model for model,_ in checkpoints)


def evaluate(rt,route):
    root,device=Path(rt['run_root']),'cuda:0'
    binding=source_control(rt,route,device)
    score(rt,route,panel=panel,modes=MODES,render=render,binding=binding,
          frozen_models=load_arms(root,binding[2],device),
          require_clean_bypass=False,replay_fn=receive_only,finalize=freeze,reader_chunk=144)


def reserve(rt):
    def bind(root,contract,device,reference,tracker):
        return partial(render,_unused=None,reference=reference,tracker=tracker,models=load_arms(root,contract,device))
    predict_clean_reserve(rt,modes=MODES,bind=bind)


@torch.no_grad()
def smoke(rt):
    root,device = Path(rt['run_root']),'cuda:0'
    (training,cal,_),contract=setup(rt,device,smoke=True)
    _,_,bound,reference,tracker=source_control(rt,'full',device)
    if bound!=contract:
        raise ValueError('Source restoration input and frozen Tri-set contracts differ')
    models=[recipe.model().to(device).eval() for recipe in RECIPES.values()]
    for model in models:
        calibrate(model,tracker,training)
    rng=torch.Generator(device=device).manual_seed(2026114451)
    clean=cal[:2]
    corrupt=real_iq(colored_observation(complex_iq(cal),rng))
    power=corruption_score(coordinates(tracker,corrupt)[0])
    chosen=corrupt[power>THRESHOLD][:2]
    if len(chosen)!=2:
        raise ValueError('Expected two active U smoke examples')
    batch=torch.cat((clean,chosen))
    actual=guarded_coordinates(tracker,batch)
    original=coordinates(tracker,batch)
    if not torch.equal(actual[1][:2],original[1][:2]):
        raise ValueError('Guarded train coordinates changed a clean smoke bypass')
    torch.testing.assert_close(actual[0]*actual[1]*actual[3],complex_iq(batch),rtol=1e-5,atol=1e-6)
    restored,active=render(batch,None,reference,tracker,models)
    logits=load_reader(root,contract,device)(restored.flatten(0,1))[0]
    if (active!=len(chosen) or not torch.equal(restored[2:,:2],clean.expand_as(restored[2:,:2]))
            or logits.shape!=(len(MODES)*len(batch),10) or not torch.isfinite(logits).all()):
        raise ValueError('Guarded slope real checkpoint identity or prediction smoke failed')
    write(root/'smoke.json',dict(status='PASS',source_U=len(batch),active=active,threshold=THRESHOLD,target_reads=0))
    print('LORA_GATED_SLOPE_ACTUAL_CHECKPOINT_PASS',flush=True)


@torch.no_grad()
def timing(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    models=load_arms(root,contract,device)
    tracker=PacketRestorer('consensus').cuda().eval()
    rng=torch.Generator(device=device).manual_seed(2026114451)
    corrupt=real_iq(colored_observation(complex_iq(cal),rng))
    pick=corruption_score(coordinates(tracker,corrupt)[0]).argmax().view(1)
    data={}
    for index,(name,model) in enumerate(zip(RECIPES,models,strict=True)):
        results={}
        for label,iq in (('clean',cal[:1]),('generic',corrupt[pick])):
            durations=[]
            for i in range(32):
                torch.cuda.synchronize()
                begin=perf_counter()
                frame=coordinates(tracker,iq)
                active=corruption_score(frame[0])>THRESHOLD
                if active.any():
                    selected=tuple(v[active] for v in frame)
                    if index==1:
                        selected=selected_pooled_frame(tracker,iq[active],selected)
                    restore(iq,active,model,selected,(.5,))
                torch.cuda.synchronize()
                if i>=8:
                    durations.append((perf_counter()-begin)*1000)
            results[label]=dict(median_ms=float(np.median(durations)),active=int(active.sum()))
        data[name]=dict(parameters=sum(p.numel() for p in model.parameters()),timing=results)
    write(root/'b1.json',dict(gpu=torch.cuda.get_device_name(),n_measurements=24,
          scope='one half-dose restoration, excludes DURA and F',arms=data))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    launch=partial(start,module=rt['runner'])
    try:
        for action,names in (('smoke',('source',)),('fit',RECIPES),('timing',('source',)),
                             ('evaluate',ROUTES),('evaluate',('reserve',))):
            schedule(rt,[(action,name) for name in names],launch)
        if launch(rt,'score','source').wait():
            raise RuntimeError('Independent LoRa guarded slope scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root/'state.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    actions=cli.add_subparsers(dest='action',required=True)
    release=actions.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        release.add_argument('--'+name,type=Path,required=True)
    for action in ('dispatch','smoke','fit','timing','evaluate','score'):
        cmd=actions.add_parser(action)
        cmd.add_argument('runtime',type=Path)
        if action!='dispatch':
            cmd.add_argument('row',choices=RECIPES if action=='fit' else
                              (*ROUTES,'reserve') if action=='evaluate' else ('source',))
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
    if action=='fit':
        name=args['row']
        fit(rt,recipe=RECIPES[name],receiver_batches=True,
            calculate=partial(natural_objective,preserve=.35,tail_weight=TAIL_WEIGHT,
                              make_pair=partial(paired_action_inputs,echo_probability=.5),
                              observation_frame=guarded_coordinates if name=='pooled' else coordinates))
    elif action=='smoke':
        smoke(rt)
    elif action=='timing':
        timing(rt)
    elif args['row']=='reserve':
        reserve(rt)
    else:
        evaluate(rt,args['row'])


if __name__ == '__main__':
    main()
