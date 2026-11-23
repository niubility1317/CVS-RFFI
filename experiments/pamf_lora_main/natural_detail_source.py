"""Train and compare weak-detail recovery using source IQ with matched noise."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score, generic_observation
from .channel_blind_source import FitRecipe, fit, panel, proof, receive_only
from .complex_vector_source import publish, score
from .denoise_source import normalized
from .dura_repair import real_iq
from .joint_unroll_source import coordinates, source_control
from .natural_detail import natural_objective, paired_echo_inputs, paired_inputs
from .packet_runner import schedule
from .pilot_joint import PilotJoint
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-natural-detail-source-m4-r01'
ECHO_RUN = '20261123-phase1-lora-dura-natural-detail-echo-source-m4-r01'
WEIGHTS = {'waveform': 0., 'detail': .35}
RECIPES = {arm: FitRecipe(arm,1800,partial(PilotJoint,equalize=False),generic_observation,PilotJoint.forward)
           for arm in WEIGHTS}
ECHO_RECIPES = {arm: FitRecipe('echo_' + arm, recipe.steps, recipe.model, recipe.perturb, recipe.restore)
                for arm, recipe in RECIPES.items()}
MODES = ('direct','reference','waveform','waveform_half','detail','detail_half')


def recipes(rt):
    run = Path(rt['run_root']).name
    if run == ECHO_RUN:
        return ECHO_RECIPES, paired_echo_inputs
    if run == RUN:
        return RECIPES, paired_inputs
    raise ValueError('Expected a registered natural-detail output root')


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference,_ = baseline(iq)
    observed,rotation,quality,rms = coordinates(tracker,iq)
    active = corruption_score(observed) > threshold
    outputs = [iq.clone() for _ in MODES[2:]]
    if active.any():
        y,rot,q,scale = (v[active] for v in (observed,rotation,quality,rms))
        for index,model in enumerate(models):
            _, restored = model(y,q,rot)
            for j,dose in enumerate((1., .5)):
                edit = torch.lerp(y,restored,dose)
                outputs[2*index+j][active] = normalized(real_iq(edit*rot*scale),iq[active])
    return torch.stack((iq,reference,*outputs)),int(active.sum())


def evaluate(rt, route):
    device = 'cuda:0'
    binding = source_control(rt,route,device)
    root = Path(rt['run_root'])
    models,thresholds = [],[]
    arms, _ = recipes(rt)
    for recipe in arms.values():
        saved = torch.load(root/f'{recipe.name}-model.pt',map_location=device,weights_only=True)
        if saved['proof'] != proof(binding[2],recipe):
            raise ValueError('The source detail model was not frozen under the matching contract')
        model = recipe.model().to(device).eval().requires_grad_(False)
        model.load_state_dict(saved['weights'],strict=True)
        models.append(model)
        thresholds.append(saved['threshold'])
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=thresholds[0]),
          binding=binding,frozen_models=tuple(models),require_clean_bypass=False,replay_fn=receive_only)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    runner = partial(start,module=rt['runner'])
    training = dict(rt,allowed_gpus=[0,4,5,1,2,3,6])
    arms, _ = recipes(rt)
    try:
        for action in ('smoke','fit'):
            write(state,dict(status=action.upper()),exclusive=False)
            schedule(training,[(action,arm) for arm in arms],runner)
        write(state,dict(status='SCORE'),exclusive=False)
        schedule(rt,[('score',route) for route in ROUTES],runner)
        write(state,dict(status='SOURCE_SCORED'),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action',required=True)
    pub = commands.add_parser('publish')
    for k in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+k,type=Path,required=True)
    pub.add_argument('--shared-echo',action='store_true')
    for action in ('dispatch','smoke','fit','score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime',type=Path)
        if action!='dispatch':
            entry.add_argument('name',choices=ROUTES if action=='score' else RECIPES)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action=='publish':
        run = ECHO_RUN if args.pop('shared_echo') else RUN
        publish(**args,run=run,runner=__spec__.name)
    else:
        rt = read(args['runtime'])
        if action=='dispatch':
            dispatch(rt)
        else:
            activate(rt)
            if action=='score':
                evaluate(rt,args['name'])
            else:
                name = args['name']
                arms, pair = recipes(rt)
                fit(rt,smoke=action=='smoke',recipe=arms[name],receiver_batches=True,
                    calculate=partial(natural_objective,preserve=WEIGHTS[name],make_pair=pair))
