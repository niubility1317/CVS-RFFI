"""Train one clean-source recovery model and freeze it before practical scoring."""

import argparse
from functools import partial
from pathlib import Path

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


def proof(contract):
    return dict(status='SOURCE_FROZEN',contract=contract,seed=SEED,steps=STEPS,
                fit_per_rx=FIT,calibration_per_rx=CAL,held_per_rx=HELD,**ISOLATION)


def setup(rt, device, *, smoke=False):
    source = Path(rt['source_root'])
    contract = source_contract(source)
    _,rx,_,_ = source_role(source,contract,'U_s')
    raw = Path(rt['parent_root']).parent/DATA_RUN/'source_views'/'U_s.npy'
    values = np.load(raw,mmap_mode='r',allow_pickle=False)
    if values.shape!=(1,len(rx),6,2,256) or values.dtype!=np.float32:
        raise ValueError('Blind training requires the single clean source U view')
    indices = partition(rx,5,4,3) if smoke else partition(rx)
    return tuple(tensor(values[0,i],device) for i in indices),contract


def waveform_risk(output, target):
    reference = torch.fft.fft(target,norm='ortho')[...,1:]
    error = torch.fft.fft(output-target,norm='ortho')[...,1:]
    power = reference.abs().square()
    return (error.abs().square()/(power+.03*power.mean(-1,keepdim=True)+.001)).log1p().mean()


def costs(model, tracker, iq, generator, *, with_identity):
    clean = complex_iq(iq)
    y = torch.stack((generic_observation(clean,generator),generic_observation(clean,generator)),dim=1)
    if with_identity:
        y[-2:] = clean[-2:,None]
    flat = real_iq(y.flatten(0,1))
    observed,rotation,q,rms = coordinates(tracker,flat)
    basis = rotation*rms
    target = clean[:,None].expand_as(y).flatten(0,1)/basis
    coarse,refined = model(observed,q)
    physical_error = ((refined-target)*basis).unflatten(0,y.shape[:2])
    paired = (physical_error[:,0]-physical_error[:,1]).abs().square().mean()
    return torch.stack((waveform_risk(coarse,target),waveform_risk(refined,target),paired))


def fit(rt, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(SEED)
    (training,calibration,held), contract = setup(rt,device,smoke=smoke)
    tracker = PacketRestorer('consensus').to(device).eval()
    model = ChannelBlind().to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    steps = 1 if smoke else STEPS
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-5)
    rng = torch.Generator(device=device).manual_seed(SEED)
    for step in range(1,steps+1):
        ix = torch.randint(len(training),(BATCH,),device=device,generator=rng)
        values = costs(model,tracker,training[ix],rng,with_identity=True)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite channel-blind source fit')
        opt.zero_grad(set_to_none=True)
        (.35*values[0]+.65*values[1]+.10*values[2]).backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step==1 and grad<=0:
            raise ValueError('No gradient on raw source IQ')
        opt.step()
        lr.step()
        if step==1 or step%50==0:
            print('CHANNEL_BLIND_FIT',step,*values.detach().tolist(),flush=True)
    with torch.no_grad():
        model.eval()
        clean_scores = torch.cat([corruption_score(coordinates(tracker,v)[0])
                                  for v in calibration.split(32)])
        threshold = float(1.05*clean_scores.max())
        eval_rng = torch.Generator(device=device).manual_seed(SEED+5)
        validation = [len(v)*costs(model,tracker,v,eval_rng,with_identity=False) for v in held.split(BATCH)]
        result = (torch.stack(validation).sum(0)/len(held)).tolist()
    root = Path(rt['run_root'])
    meta = proof(contract)
    summary = {k:v for k,v in meta.items() if k!='contract'}
    summary.update(threshold=threshold,held_waveform=result,
                   n_fit=len(training),n_calibration=len(calibration),n_held=len(held))
    if smoke:
        write(root/'smoke-blind.json',dict(summary,status='PASS',steps=1))
    else:
        torch.save(dict(proof=meta,weights=model.state_dict(),threshold=threshold),root/'blind-model.pt')
        write(root/'blind-complete.json',summary)


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
