"""Run parallel joint recovery with and without observable pilot equalization."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, proof, receive_only
from .complex_vector_source import publish, score
from .denoise_source import normalized
from .dura_repair import real_iq
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .pilot_joint import PilotJoint, generic_distortion, pilot_inverse
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_signal_probe import observed_noise_rms


RUN = '20261123-phase1-lora-dura-pilot-joint-source-m4-r01'
STAGED_RUN = '20261123-phase1-lora-dura-pilot-stage-schedule-source-m4-r01'
ARMS = ('identity','pilot')
MODES = ('direct','reference','pilot_only','identity_coarse','identity_joint',
         'pilot_coarse','pilot_joint','pilot_joint_half')


def restore(model,y,q,template):
    return model(y,q,template)


def recipes(rt):
    staged = rt['run_id'] == STAGED_RUN
    return {a: FitRecipe(a, 1400, partial(PilotJoint, equalize=a=='pilot'),
                        generic_distortion, restore, coarse_first=staged) for a in ARMS}


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference,_ = baseline(iq)
    observed,rotation,quality,rms = coordinates(tracker,iq)
    active = corruption_score(observed)>threshold
    outputs = [iq.clone() for _ in MODES[2:]]
    if active.any():
        y,rot,q,s = (v[active] for v in (observed,rotation,quality,rms))
        physics = pilot_inverse(y,y,rot,observed_noise_rms(y).square())
        a,b = (m(y,q,rot) for m in models)
        restored = physics,*a,*b,.5*(y+b[-1])
        for result,value in zip(outputs,restored,strict=True):
            result[active] = normalized(real_iq(value*rot*s),iq[active])
    return torch.stack((iq,reference,*outputs)),int(active.sum())


def evaluate(rt, route):
    device = 'cuda:0'
    binding = source_control(rt,route,device)
    root = Path(rt['run_root'])
    models, thresholds = [], []
    for recipe in recipes(rt).values():
        saved = torch.load(root/f'{recipe.name}-model.pt',map_location=device,weights_only=True)
        if saved['proof']!=proof(binding[2],recipe):
            raise ValueError('The matched joint raw source models must be frozen')
        model = recipe.model().to(device).eval().requires_grad_(False)
        model.load_state_dict(saved['weights'],strict=True)
        models.append(model)
        thresholds.append(saved['threshold'])
    # Use the preregistered identity calibration for every readout.
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=thresholds[0]),
          binding=binding,frozen_models=tuple(models),require_clean_bypass=False,replay_fn=receive_only)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    runner = partial(start,module=rt['runner'],extra_keys=('run_id',))
    training = dict(rt,allowed_gpus=[0,4,5,1,2,3,6])
    try:
        for action in ('smoke','fit'):
            write(state,dict(status=action.upper()),exclusive=False)
            schedule(training,[(action,a) for a in ARMS],runner)
        write(state,dict(status='SCORE'),exclusive=False)
        schedule(rt,[('score',r) for r in ROUTES],runner)
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
    pub.add_argument('--staged',action='store_true')
    for action in ('dispatch','smoke','fit','score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime',type=Path)
        if action!='dispatch':
            entry.add_argument('name',choices=ROUTES if action=='score' else ARMS)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action=='publish':
        run = STAGED_RUN if args.pop('staged') else RUN
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
                fit(rt,smoke=action=='smoke',recipe=recipes(rt)[args['name']])
