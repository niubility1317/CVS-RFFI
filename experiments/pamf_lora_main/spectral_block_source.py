"""Train matched LoRa spectral-block priors and score fixed source V decisions."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, proof, receive_only, setup
from .complex_vector_source import load_reader, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_publish import DATA_RUN
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import ROUTES, freeze, score_frozen
from .natural_detail import natural_objective
from .publish import release, release_commit
from .refine_runner import select_gpu, start
from .residual_dura import RUN as RESIDUAL_RUN
from .runner import activate
from .spectral_block_recovery import SpectralBlockRecovery, varied_observation


RUN = '20261123-phase1-lora-spectral-block-source-m4-r01'
STEPS = 1600
DETAIL_WEIGHT = .35
MODES = ('direct','reference','unmasked_coarse','unmasked_full','unmasked_half',
         'masked_coarse','masked_full','masked_half')
COMPARISONS = (('masked_coarse','unmasked_coarse'), ('masked_full','unmasked_full'),
               ('masked_half','unmasked_half'), ('masked_full','masked_coarse'),
               ('masked_half','direct'), ('unmasked_half','direct'))


def restore(model, y, quality, _rotation):
    return model(y, quality)


RECIPES = {arm: FitRecipe(arm, STEPS, partial(SpectralBlockRecovery, masked=masked),
                          varied_observation, restore)
           for arm, masked in (('unmasked', False), ('masked', True))}


def load_models(root, contract, device, recipes=RECIPES):
    models, thresholds = [], []
    for arm, recipe in recipes.items():
        cp = torch.load(root/f'{arm}-model.pt', map_location=device, weights_only=True)
        if cp['proof'] != proof(contract, recipe):
            raise ValueError('Need the frozen same-contract U-only recovery arm')
        model = recipe.model().to(device).eval().requires_grad_(False)
        model.load_state_dict(cp['weights'], strict=True)
        models.append(model)
        thresholds.append(cp['threshold'])
    if not thresholds or not np.isfinite(thresholds[0]) or thresholds[0] <= 0:
        raise ValueError('Expected a finite positive source U threshold')
    if any(value != thresholds[0] for value in thresholds[1:]):
        raise ValueError('Source U threshold must be identical across arms')
    return tuple(models), thresholds[0]


def recover(iq, tracker, models, threshold):
    y, rotation, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y)>threshold
    out = [iq.clone() for _ in range(3*len(models))]
    if active.any():
        z, q, base = y[active], quality[active], rotation[active]*rms[active]
        received = iq[active]
        for i, model in enumerate(models):
            first, fine = model(z,q)
            correction = real_iq((fine-z)*base)
            out[3*i][active] = received+real_iq((first-z)*base)
            out[3*i+1][active] = received+correction
            out[3*i+2][active] = received+.5*correction
    return out, int(active.sum())


def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    proposals, active = recover(iq, tracker, models, threshold)
    return torch.stack((iq,reference,*proposals)), active


@torch.no_grad()
def smoke(rt, recipes=RECIPES, modes=MODES):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    (_, source, _), contract = setup(rt, device, smoke=True)
    models, threshold = load_models(root, contract, device, recipes)
    model = models[1]
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('The calibration and reference must share the source contract')
    reader = load_reader(root, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026112329)
    disturbed = real_iq(varied_observation(complex_iq(source), rng))
    ratio = corruption_score(coordinates(tracker, disturbed)[0])
    x = disturbed[ratio.argmax().view(1)]
    waves, active = render(torch.cat((source[:1],x)), None, baseline, tracker, models, threshold=threshold)
    logits = reader(waves.flatten(0,1))[0]
    if logits.shape != (2*len(modes),10) or not torch.isfinite(logits).all() or not active:
        raise ValueError('Expected real frozen Tri-set, recovery and DURA outputs')
    times = {}
    for name, sample in (('source_clean', source[:1]), ('generic_disturbed', x)):
        records = []
        for step in range(28):
            torch.cuda.synchronize()
            begun = perf_counter()
            _, enable = recover(sample, tracker, (model,), threshold)
            torch.cuda.synchronize()
            if step >= 8:
                records.append(1000*(perf_counter()-begun))
        times[name] = dict(active=bool(enable), median_ms=float(np.median(records)))
    write(root/'smoke.json', dict(status='PASS', target_reads=0, V_reads=0,
          parameters={name:sum(p.numel() for p in m.parameters()) for name,m in zip(recipes,models,strict=True)},
          b1=times, device=torch.cuda.get_device_name()))


def evaluate(rt, route, recipes=RECIPES, modes=MODES):
    binding = source_control(rt, route, 'cuda:0')
    models, threshold = load_models(Path(rt['run_root']), binding[2], 'cuda:0', recipes)
    score(rt, route, panel=panel, modes=modes, render=partial(render, threshold=threshold),
          binding=binding, frozen_models=models, require_clean_bypass=False, replay_fn=receive_only, finalize=freeze)


def paired_tasks(rt, kind, names):
    slots = []
    for _ in names:
        slots.append(select_gpu(exclude=slots))
    write(Path(rt['run_root'])/'state.json', dict(status=kind.upper(), gpus=dict(zip(names,slots,strict=True))), exclusive=False)
    jobs = [start(rt,kind,name,gpu,module=rt['runner']) for name,gpu in zip(names,slots,strict=True)]
    if any([p.wait() for p in jobs]):
        raise RuntimeError(f'LoRa matched spectral-block {kind} failed')


def dispatch(rt, recipes=RECIPES):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root/'state.json'
    try:
        paired_tasks(rt, 'fit_smoke', recipes)
        paired_tasks(rt, 'fit', recipes)
        gpu = select_gpu()
        write(status, dict(status='SMOKE', gpu=gpu), exclusive=False)
        if start(rt,'smoke','source',gpu,module=rt['runner']).wait():
            raise RuntimeError('LoRa real checkpoint no-query smoke failed')
        paired_tasks(rt, 'evaluate', ROUTES)
        write(status, dict(status='SCORE'), exclusive=False)
        if start(rt,'score','source',module=rt['runner']).wait():
            raise RuntimeError('Independent LoRa matched-source scoring failed')
        write(status, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(status, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, *, run=RUN, runner=None):
    cfg = read(context)
    rt = {key:cfg[key] for key in ('project','python','frozen_sources')}
    p = PurePosixPath(rt['project'])
    rt.update(run_id=run, runner=runner or __spec__.name, code_commit=release_commit(repo),
              release_root=str(p/'releases'/f'{run}-release'), run_root=str(p/'runs'/run),
              source_root=str(p/'releases'/f'{DATA_RUN}-release'/'data'/'source'),
              parent_root=str(p/'runs'/RESIDUAL_RUN))
    release(repo, output, rt, ssh_config)


def main(*, run=RUN, runner=None, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=partial(natural_objective, preserve=DETAIL_WEIGHT), run_smoke=smoke, run_evaluate=evaluate):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','fit_smoke','fit','smoke','evaluate','score'):
        c = actions.add_parser(action)
        c.add_argument('runtime',type=Path)
        if action!='dispatch':
            c.add_argument('name',choices=recipes if action in ('fit','fit_smoke') else ROUTES if action=='evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args, run=run, runner=runner)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt, recipes)
    elif action=='score':
        score_frozen(rt, modes, comparisons=comparisons)
    else:
        activate(rt)
        if action in ('fit','fit_smoke'):
            fit(rt, smoke=action=='fit_smoke', recipe=recipes[args['name']], receiver_batches=True,
                calculate=calculate)
        elif action=='smoke':
            run_smoke(rt, recipes, modes)
        else:
            run_evaluate(rt,args['name'],recipes,modes)


if __name__=='__main__':
    main()
