"""Train matched LoRa dose and severe-observation controls on source IQ."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .adaptive_dose import AdaptiveDoseDenoise, dose_risk
from .complex_vector_source import dispatch, publish, record, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX, normalized
from .dura_repair import PacketRestorer, real_iq
from .joint_unroll_source import coordinates
from .learned_inverse_source import panel as previous_inverse
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .shared_spectrum_source import panel as previous_shared
from .signal_retention_source import build_arrays
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-adaptive-dose-source-m2-r01'
ARMS = ('fixed','adaptive','severe')
RECIPE = SimpleNamespace(seed=2026112961,steps=1900,train_per_rx=1000,held_per_rx=80,
                         arms=ARMS,model=lambda a: AdaptiveDoseDenoise(adaptive=a!='fixed'))
MODES = ('direct','reference','fixed','fixed_half','adaptive','adaptive_half',
         'severe','severe_half','known_80','known_100')
BATCH = 10


def panel(rx):
    used = np.union1d(previous_shared(rx),previous_inverse(rx,60))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx==key),used),
                                             EVAL_PER_RX,replace=False)) for key in SOURCE])


@torch.no_grad()
def held_diagnostics(model, arrays):
    received,quality,signal,_,_,_ = arrays
    totals = []
    model.eval()
    for y,q,s in zip(*(v.flatten(0,1).split(16) for v in (received,quality,signal)),strict=True):
        noise,sigma,limit = model(y,q,details=True)
        true = y-s
        e = (true-noise).abs().square().mean(-1)/true.abs().square().mean(-1).clamp_min(1e-10)
        dose = noise.abs().square().mean(-1).sqrt()
        totals.append(torch.stack((sigma[...,0],e,dose,limit[...,0]),dim=-1).flatten(0,1))
    out = torch.cat(totals)
    severe = out[:,0]>=.55
    report = {'exposures':len(received)*2,'chirps':len(out)}
    for key, data in (('all',out),('severe',out[severe])):
        if not len(data):
            continue
        cuts = data.new_tensor((.50,.90))
        report[key] = dict(n=len(data),error_p50_p90=torch.quantile(data[:,1],cuts).tolist(),
                           dose_p50_p90=torch.quantile(data[:,2],cuts).tolist(),
                           cap_rate=float((data[:,2]>data[:,3]-.0001).float().mean()))
    return report


def fit(rt, route, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(RECIPE.seed)
    tracker = PacketRestorer('consensus').to(device).eval()
    count,hold = (6,4) if smoke else (RECIPE.train_per_rx,RECIPE.held_per_rx)
    source,held,contract = build_arrays(rt,route,tracker,count,hold)
    measured,quality,exact,_,clean,clean_quality = source
    models = {a:RECIPE.model(a).to(device) for a in ARMS}
    for a in ARMS[1:]:
        models[a].load_state_dict(models[ARMS[0]].state_dict())
    opt = torch.optim.AdamW([p for m in models.values() for p in m.parameters()],
                            lr=2e-4,weight_decay=1e-4)
    steps = 1 if smoke else RECIPE.steps
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-5)
    rng = torch.Generator(device=device).manual_seed(RECIPE.seed)
    for step in range(1,steps+1):
        sample = torch.randint(len(measured),(BATCH,),device=device,generator=rng)
        y,q,s = [v[sample] for v in (measured,quality,exact)]
        y[-2:] = s[-2:] = clean[sample[-2:],None]
        q[-2:] = clean_quality[sample[-2:],None]
        y,q,s = (v.flatten(0,1) for v in (y,q,s))
        risks = []
        for model in models.values():
            noise,sigma,_ = model(y,q,details=True)
            risks.append(dose_risk(noise,y,s,sigma))
        values = torch.stack(risks)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite dose or severity objective')
        opt.zero_grad(set_to_none=True)
        (values[:2,0].sum()+values[2,1]).backward()
        for model in models.values():
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if step==1 and norm<=0:
                raise ValueError('Expected a legal nonzero source gradient')
        opt.step()
        lr.step()
        if step==1 or step%50==0:
            print('ADAPTIVE_DOSE_FIT',route,step,*values.detach().flatten().tolist(),flush=True)
    proof = record(contract,route,RECIPE)
    result = {k:v for k,v in proof.items() if k!='contract'}
    result.update(fit_packets=len(measured),held_packets=len(held[0]),
                  held={k:held_diagnostics(m,held) for k,m in models.items()})
    root = Path(rt['run_root'])
    if smoke:
        write(root/f'smoke-{route}.json',dict(result,status='PASS',steps=1))
    else:
        torch.save(dict(proof=proof,models={k:m.state_dict() for k,m in models.items()}),
                   root/f'{route}-model.pt')
        write(root/f'{route}-complete.json',result)


@torch.no_grad()
def render(iq,noise,baseline,tracker,models):
    reference,active = baseline(iq)
    outputs = [iq.clone() for _ in MODES[2:]]
    if active.any():
        observed = iq[active]
        y,phase,q,rms = coordinates(tracker,observed)
        prediction = [real_iq(m(y,q)*phase*rms) for m in models]
        updates = [value*fraction for value in prediction for fraction in (1.,.5)]
        updates += [.8*noise[active],noise[active]]
        for wave,estimate in zip(outputs,updates,strict=True):
            wave[active] = normalized(observed-estimate,observed)
    return torch.stack((iq,reference,*outputs)),int(active.sum())


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
            entry.add_argument('route',choices=ROUTES)
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
            route = args['route']
            if action=='score':
                score(rt,route,recipe=RECIPE,panel=panel,modes=MODES,render=render)
            else:
                source_trace_implementation()
                fit(rt,route,smoke=action=='smoke')
