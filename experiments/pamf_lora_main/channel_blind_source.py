"""Train one clean-source recovery model and freeze it before practical scoring."""

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .channel_blind import ChannelBlind, corruption_score, generic_observation
from .complex_vector_source import publish, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX, normalized
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates
from .packet_physics import aligned_view, chirp_windows
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-channel-blind-source-m3-r01'
SEED, STEPS, BATCH = 2026112991, 1900, 12
FIT, CAL, HELD = 1000, 500, 100
MODES = ('direct','reference','blind_coarse','blind_refined','blind_half')
ISOLATION = dict(F_reads=0,TX_reads=0,V_reads=0,target_reads=0,practical_reads=0,trace_reads=0)
JOINT_WEIGHTS = (.35, .65, .10)


@dataclass(frozen=True)
class FitRecipe:
    name: str
    steps: int
    model: Callable
    perturb: Callable
    restore: Callable
    coarse_first: bool = False
    batch: int = BATCH
    fit_per_rx: int = FIT
    validation_steps: tuple[int, ...] = ()
    bank_per_rx: int = 0
    initialize: Callable | None = None

    def loss_weights(self, step):
        if not self.coarse_first:
            return JOINT_WEIGHTS
        start, end = self.steps // 5, self.steps // 2
        progress = max(0., min(1., (step - start) / max(1, end - start)))
        initial = (.80, .20, .02)
        return tuple((1. - progress) * low + progress * high
                     for low, high in zip(initial, JOINT_WEIGHTS, strict=True))


DEFAULT = FitRecipe('blind',STEPS,ChannelBlind,generic_observation,
                    lambda model,y,q,_template:model(y,q))


def partition(rx, fit=FIT, cal=CAL, held=HELD):
    rng = np.random.default_rng(SEED)
    parts = [rng.permutation(np.flatnonzero(rx==key)) for key in SOURCE]
    if any(len(r)<fit+cal+held for r in parts):
        raise ValueError('Need distinct raw source U fit/clean calibration/held roles')
    return tuple(np.concatenate([r[a:b] for r in parts])
                 for a,b in [(0,fit),(fit,fit+cal),(fit+cal,fit+cal+held)])


def panel(rx):
    rng = np.random.default_rng(SEED+1)
    return np.concatenate([np.sort(rng.choice(np.flatnonzero(rx==key),EVAL_PER_RX,
                                             replace=False)) for key in SOURCE])


def proof(contract, recipe=DEFAULT):
    metadata = dict(status='SOURCE_FROZEN',contract=contract,seed=SEED,steps=recipe.steps,
                fit_per_rx=recipe.fit_per_rx,calibration_per_rx=CAL,held_per_rx=HELD,**ISOLATION)
    if recipe.batch != BATCH:
        metadata['batch'] = recipe.batch
    if recipe.bank_per_rx:
        metadata['bank_per_rx'] = recipe.bank_per_rx
    if recipe.name!='blind':
        metadata['estimator'] = recipe.name
    if recipe.coarse_first:
        metadata['loss_schedule'] = 'coarse_first'
    return metadata


def setup(rt, device, *, smoke=False, fit_per_rx=FIT, bank_per_rx=0):
    source = Path(rt['source_root'])
    contract = source_contract(source)
    _,rx,_,_ = source_role(source,contract,'U_s')
    raw = Path(rt['parent_root']).parent/DATA_RUN/'source_views'/'U_s.npy'
    values = np.load(raw,mmap_mode='r',allow_pickle=False)
    if values.shape!=(1,len(rx),6,2,256) or values.dtype!=np.float32:
        raise ValueError('Blind training requires the single clean source U view')
    reserved = bank_per_rx
    indices = partition(rx,5+reserved,4,3) if smoke else partition(rx,fit=fit_per_rx+reserved)
    if reserved:
        fit, cal, held = indices
        indices = (fit.reshape(len(SOURCE),-1)[:,reserved:].reshape(-1),cal,held)
    return tuple(tensor(values[0,i],device) for i in indices),contract


def waveform_risk(output, target):
    reference = torch.fft.fft(target,norm='ortho')[...,1:]
    error = torch.fft.fft(output-target,norm='ortho')[...,1:]
    power = reference.abs().square()
    return (error.abs().square()/(power+.03*power.mean(-1,keepdim=True)+.001)).log1p().mean()


@torch.no_grad()
def clean_threshold(tracker, calibration):
    scores = [corruption_score(coordinates(tracker, batch)[0]) for batch in calibration.split(32)]
    return 1.05 * torch.cat(scores).max()


def costs(model, tracker, iq, generator, *, with_identity, recipe=DEFAULT):
    clean = complex_iq(iq)
    y = torch.stack((recipe.perturb(clean,generator),recipe.perturb(clean,generator)),dim=1)
    if with_identity:
        y[-2:] = clean[-2:,None]
    flat = real_iq(y.flatten(0,1))
    observed,rotation,q,rms = coordinates(tracker,flat)
    basis = rotation*rms
    target = clean[:,None].expand_as(y).flatten(0,1)/basis
    coarse,refined = recipe.restore(model,observed,q,rotation)
    physical_error = ((refined-target)*basis).unflatten(0,y.shape[:2])
    paired = (physical_error[:,0]-physical_error[:,1]).abs().square().mean()
    return torch.stack((waveform_risk(coarse,target),waveform_risk(refined,target),paired))


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    values = costs(model, tracker, iq, rng, with_identity=with_identity, recipe=recipe)
    coarse, refined, paired = recipe.loss_weights(step)
    loss = coarse * values[0] + refined * values[1] + paired * values[2]
    return loss, values


@torch.no_grad()
def held_waveform(model, tracker, held, calculate, recipe, step, receiver_batches):
    rng = torch.Generator(device=held.device).manual_seed(SEED+5)
    groups = held.chunk(len(SOURCE)) if receiver_batches else (held,)
    model.eval()
    values = [len(x)*calculate(model,tracker,x,rng,with_identity=False,recipe=recipe,step=step)[1]
              for group in groups for x in group.split(recipe.batch)]
    result = torch.stack(values).sum(0)/len(held)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite held source waveform risk')
    return result.tolist()


def fit(rt, *, smoke=False, recipe=DEFAULT, calculate=objective, receiver_batches=False):
    device = 'cuda:0'
    torch.manual_seed(SEED)
    (training,calibration,held), contract = setup(rt,device,smoke=smoke,fit_per_rx=recipe.fit_per_rx,
                                                   bank_per_rx=recipe.bank_per_rx)
    tracker = PacketRestorer('consensus').to(device).eval()
    model = recipe.model().to(device)
    if recipe.initialize is not None:
        recipe.initialize(model, tracker, training, calibration)
    opt = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    steps = 1 if smoke else recipe.steps
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-5)
    rng = torch.Generator(device=device).manual_seed(SEED)
    start_time = time.perf_counter()
    progress = {}
    for step in range(1,steps+1):
        if receiver_batches:
            count = len(training) // len(SOURCE)
            receiver = torch.randint(len(SOURCE),(1,),device=device,generator=rng)
            ix = receiver*count + torch.randint(count,(recipe.batch,),device=device,generator=rng)
        else:
            ix = torch.randint(len(training),(recipe.batch,),device=device,generator=rng)
        loss,values = calculate(model,tracker,training[ix],rng,with_identity=True,
                                recipe=recipe,step=step)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite channel-blind source fit')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step==1 and grad<=0:
            raise ValueError('No gradient on raw source IQ')
        opt.step()
        lr.step()
        if step==1 or step%50==0:
            print('CHANNEL_BLIND_FIT',step,*values.detach().tolist(),flush=True)
        if not smoke and step in recipe.validation_steps:
            result = held_waveform(model,tracker,held,calculate,recipe,step,receiver_batches)
            progress[str(step)] = result
            print('CHANNEL_BLIND_HELD',step,*result,flush=True)
            model.train()
    torch.cuda.synchronize()
    worker_seconds = time.perf_counter()-start_time
    with torch.no_grad():
        model.eval()
        threshold = float(clean_threshold(tracker, calibration))
        result = held_waveform(model,tracker,held,calculate,recipe,steps,receiver_batches)
    root = Path(rt['run_root'])
    meta = proof(contract,recipe)
    summary = {k:v for k,v in meta.items() if k!='contract'}
    summary.update(threshold=threshold,held_waveform=result,
                   n_fit=len(training),n_calibration=len(calibration),n_held=len(held),
                   worker_seconds=worker_seconds,
                   bf16_training_convolutions=recipe.name!='blind' and torch.cuda.is_bf16_supported())
    if progress:
        summary['held_trajectory'] = progress
        summary['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    if smoke:
        write(root/f'smoke-{recipe.name}.json',dict(summary,status='PASS',steps=1))
    else:
        torch.save(dict(proof=meta,weights=model.state_dict(),threshold=threshold),root/f'{recipe.name}-model.pt')
        write(root/f'{recipe.name}-complete.json',summary)


def receive_only(route, scene, row):
    iq,rx,key = row
    y = chirp_windows(iq) if scene=='clean' else aligned_view(iq,scene,rx,key,'source-V',processing_route=route)
    return y,np.zeros_like(y)


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference,_ = baseline(iq)
    observed,phase,q,rms = coordinates(tracker,iq)
    active = corruption_score(observed)>threshold
    output = [iq.clone() for _ in MODES[2:]]
    if active.any():
        coarse,refined = models[0](observed[active],q[active])
        midpoint = .5*(observed[active]+refined)
        for row,value in zip(output,(coarse,refined,midpoint),strict=True):
            restored = real_iq(value*phase[active]*rms[active])
            row[active] = normalized(restored,iq[active])
    return torch.stack((iq,reference,*output)),int(active.sum())


def evaluate(rt, route):
    from .joint_unroll_source import source_control
    device = 'cuda:0'
    binding = source_control(rt,route,device)
    folder = Path(rt['run_root'])
    saved = torch.load(folder/'blind-model.pt',map_location=device,weights_only=True)
    if saved['proof']!=proof(binding[2]):
        raise ValueError('Blind model is not bound to the same raw source role')
    model = ChannelBlind().to(device).eval().requires_grad_(False)
    model.load_state_dict(saved['weights'],strict=True)
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=saved['threshold']),
          binding=binding,frozen_models=(model,),require_clean_bypass=False,replay_fn=receive_only)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    runner = partial(start,module=rt['runner'])
    training = dict(rt,allowed_gpus=[2,0,4,1,5,3,6])
    try:
        for action in ('smoke','fit'):
            write(state,dict(status=action.upper()),exclusive=False)
            schedule(training,[(action,'blind')],runner)
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
    for option in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+option,type=Path,required=True)
    for action in ('dispatch','smoke','fit','score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime',type=Path)
        if action!='dispatch':
            entry.add_argument('route',choices=('blind',) if action!='score' else ROUTES)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args,run=RUN,runner=__spec__.name)
    else:
        rt = read(args['runtime'])
        if action=='dispatch':
            dispatch(rt)
        else:
            activate(rt)
            if action=='score':
                evaluate(rt,args['route'])
            else:
                fit(rt,smoke=action=='smoke')
