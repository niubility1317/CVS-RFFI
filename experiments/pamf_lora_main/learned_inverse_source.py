"""Measure known-action inversion after an actual frozen LoRa denoiser."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .complex_vector_source import load_models, publish
from .data import SOURCE, VIEWS
from .denoise_source import normalized, replay
from .dura_repair import real_iq
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_packet_views import source_role
from .full_supervised import construct as recognizer
from .inverse_noise_dose_source import panel as dose_panel
from .joint_unroll_source import coordinates, source_control
from .physical_inverse_source import dispatch, source_trace_implementation, summarize
from .registered_inverse import inverse_row
from .residual_dura import SEED
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_signal_probe_source import RECIPE, RUN as PARENT, panel as weak_panel
from .weak_signal_probe import observed_noise_rms


RUN = '20261123-phase1-lora-dura-learned-inverse-source-m2-r01'
PER_RX = 60
MODES = ('direct', 'reference', 'learned', 'learned_half', 'known_80', 'known_100',
         'inverse_raw', 'inverse_learned', 'inverse_learned_half', 'inverse_80', 'inverse_100')


def panel(rx, count):
    used = np.union1d(weak_panel(rx),dose_panel(rx,PER_RX))
    rng = np.random.default_rng(2026112931)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx==key),used),
                                             count,replace=False)) for key in SOURCE])


@torch.no_grad()
def input_readouts(received, exact, baseline, tracker, denoiser):
    reference, active = baseline(received)
    guess = torch.zeros_like(received)
    noise_rms = torch.zeros_like(received[...,0,0])
    if active.any():
        y, rotation, q, rms = coordinates(tracker,received[active])
        guess[active] = real_iq(denoiser(y,q)*rotation*rms)
        noise_rms[active] = (observed_noise_rms(y)*rms)[...,0]
    selected = active[:,None,None,None]
    guesses = (guess,.5*guess,.80*exact,exact)
    values = [torch.where(selected,normalized(received-g,received),received) for g in guesses]
    return torch.stack((received,reference,*values)),active,guess,noise_rms


@torch.no_grad()
def evaluate(rt, route, *, smoke=False):
    root = Path(rt['run_root'])
    device = 'cuda:0'
    source, cache, contract, baseline, tracker = source_control(rt,route,device)
    parent = root.parent/PARENT
    if read(parent/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Need the completed matching source denoiser')
    _, denoiser = load_models(parent,route,contract,RECIPE,device)
    raw, rx, ids, _ = source_role(source,contract,'V')
    chosen = panel(rx,2 if smoke else PER_RX)
    rows = list(zip(raw[chosen],rx[chosen],ids[chosen],strict=True))
    official = None if cache is None else np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    if not smoke:
        f = recognizer(device).eval().requires_grad_(False)
        weight = complete_f(root.parent/SET_RUN,contract,case(SEED,SET_METHOD))/'model.pt'
        f.load_state_dict(torch.load(weight,map_location=device,weights_only=True),strict=True)
    metrics, prediction = {}, {}
    views = ('clean','practical_low_urban') if smoke else VIEWS
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in views:
            captures, noise = (np.stack(v) for v in zip(*pool.map(partial(replay,route,scene),rows,
                                                              chunksize=4),strict=True))
            if official is not None and not np.array_equal(captures,official[VIEWS.index(scene),chosen]):
                raise ValueError('Source full IQ must match the registered forward replay')
            prepared = [input_readouts(y,n,baseline,tracker,denoiser)
                        for y,n in zip(tensor(captures,device).split(24),
                                       tensor(noise,device).split(24),strict=True)]
            direct = torch.cat([p[0] for p in prepared],dim=1)
            active = torch.cat([p[1] for p in prepared]).cpu().tolist()
            guess = torch.cat([p[2] for p in prepared]).cpu()
            noise_rms = torch.cat([p[3] for p in prepared]).cpu().tolist()
            use = np.flatnonzero(active)
            inverse = direct[0].unsqueeze(0).repeat(5,1,1,1,1)
            residual = 0.
            if len(use):
                jobs = ((*rows[i],captures[i],np.asarray(guess[i].tolist(),dtype=np.float32),
                         np.asarray(noise_rms[i],dtype=np.float64))
                        for i in use)
                solutions, errors = zip(*pool.map(partial(inverse_row,route,scene),jobs,chunksize=2),
                                         strict=True)
                inverse[:,torch.as_tensor(use.tolist(),device=device)] = (
                    tensor(np.stack(solutions,axis=1),device))
                residual = max(errors)
            outputs = torch.cat((direct,inverse),dim=0)
            if not torch.isfinite(outputs).all() or (scene=='clean' and len(use)):
                raise ValueError('Nonfinite projected inverse or invalid clean bypass')
            metrics[scene] = dict(active=len(use),replay_relative_error=residual)
            print('REGISTERED_INVERSE_SOURCE',route,scene,len(use),residual,flush=True)
            if not smoke:
                logits = [f(group.flatten(0,1))[0] for group in outputs.split(18,dim=1)]
                if not all(torch.isfinite(x).all() for x in logits):
                    raise FloatingPointError('Nonfinite frozen predictions')
                parts = [x.argmax(1).reshape(len(MODES),-1).cpu().tolist() for x in logits]
                prediction[scene] = np.concatenate(parts,axis=1)
    if smoke:
        if not metrics['practical_low_urban']['active']:
            raise ValueError('Source low-urban samples did not exercise the inverse')
        write(root/f'smoke-{route}.json',dict(status='PASS',route=route,metrics=metrics,
              F_reads=0,TX_reads=0,target_reads=0))
    else:
        summarize(root,route,source,ids,chosen,prediction,metrics,MODES)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action',required=True)
    pub = commands.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for action in ('dispatch','smoke','score'):
        command = commands.add_parser(action)
        command.add_argument('runtime',type=Path)
        if action!='dispatch':
            command.add_argument('route',choices=ROUTES)
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
            source_trace_implementation()
            evaluate(rt,args['route'],smoke=action=='smoke')
