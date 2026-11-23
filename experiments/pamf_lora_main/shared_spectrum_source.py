"""Fit physical-dose LoRa common-spectrum refiners using only source IQ."""

import argparse
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .complex_vector import fft
from .complex_vector_source import dispatch, load_models, publish, record, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX, normalized
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .packet_subspace_cross_source import panel as cross_panel
from .packet_subspace_source import RECIPE as PARENT_RECIPE, RUN as PARENT, panel as parent_panel
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .shared_spectrum import SharedSpectrum, common_inputs, errors, match_common, packet_mean
from .signal_retention_source import build_arrays
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-shared-spectrum-source-m2-r01'
RECIPE = SimpleNamespace(seed=2026112913, steps=1900, train_per_rx=1000, held_per_rx=80,
                         arms=('ordinary', 'fine_bins'), model=lambda _: SharedSpectrum())
MODES = ('direct', 'reference', 'frozen_factorized', 'ordinary', 'fine_bins', 'fine_bins_half',
         'known_shared_direction', 'known_noise')
BATCH = 28


def panel(rx):
    used = np.union1d(parent_panel(rx), cross_panel(rx))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), used),
                                             EVAL_PER_RX, replace=False)) for key in SOURCE])


def parent(root, route, contract, device):
    path = root.parent/PARENT
    if read(path/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Need the completed factorized parent under the same source contract')
    return load_models(path, route, contract, PARENT_RECIPE, device)[1]


@torch.no_grad()
def pack(arrays, frozen):
    observed, quality, signal, base, gate, _, _, rms = arrays
    used = gate.flatten()
    values = [v.flatten(0, 1)[used] for v in (observed, base, quality, rms, signal)]
    if len(values[0])<BATCH:
        raise ValueError('Insufficient source observations through the fixed gate')
    rows = []
    for y, b, q, scale, target in zip(*(x.split(36) for x in values), strict=True):
        pred = frozen(y,b,q)
        features, anchor = common_inputs(y, b, pred, q, scale)
        exact = packet_mean(fft(target-b), scale)
        rows.append((features, anchor, match_common(anchor, exact)))
    return tuple(torch.cat(v) for v in zip(*rows, strict=True))


@torch.no_grad()
def hold_loss(net, hold):
    net.eval()
    rows = []
    for features, reference, exact in zip(*(x.split(BATCH) for x in hold), strict=True):
        rows.append(len(features)*errors(net(features,reference), exact))
    return (torch.stack(rows).sum(0)/len(hold[0])).tolist()


def fit(rt, route, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(RECIPE.seed)
    root = Path(rt['run_root'])
    _, _, contract, baseline, tracker = source_control(rt, route, device)
    count, hold = (8, 5) if smoke else (RECIPE.train_per_rx, RECIPE.held_per_rx)
    a, b, bound = build_arrays(rt, route, tracker, count, hold, baseline=baseline, with_rms=True)
    if contract != bound:
        raise ValueError('Mismatch between replay and fixed recovery parents')
    frozen = parent(root, route, contract, device)
    source = pack(a, frozen)
    held = pack(b, frozen)
    del a, b
    models = {k: RECIPE.model(k).to(device) for k in RECIPE.arms}
    models['fine_bins'].load_state_dict(models['ordinary'].state_dict())
    opt = torch.optim.AdamW([p for m in models.values() for p in m.parameters()],
                            lr=2e-4, weight_decay=1e-4)
    steps = 1 if smoke else RECIPE.steps
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=2e-5)
    rng = torch.Generator(device=device).manual_seed(RECIPE.seed)
    for step in range(1,steps+1):
        ix = torch.randint(len(source[0]), (BATCH,), device=device, generator=rng)
        x, ref, exact = (a[ix] for a in source)
        costs = torch.stack([errors(m(x,ref), exact) for m in models.values()])
        if not torch.isfinite(costs).all():
            raise FloatingPointError('Nonfinite source shared-spectrum reconstruction')
        opt.zero_grad(set_to_none=True)
        costs.diagonal().sum().backward()
        for net in models.values():
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            if step==1 and grad<=0:
                raise ValueError('Expected real source common direction gradients in both arms')
        opt.step()
        lr.step()
        if step==1 or step%50==0:
            print('SHARED_SPECTRUM_FIT', route, step, *costs.detach().flatten().tolist(), flush=True)
    proof = record(contract, route, RECIPE)
    result = {k:v for k,v in proof.items() if k!='contract'}
    result.update(prior=PARENT, fit_exposures=len(source[0]), held_exposures=len(held[0]),
                  frozen_errors=errors(held[1], held[2]).tolist(),
                  held={k:hold_loss(m, held) for k,m in models.items()})
    if smoke:
        write(root/f'smoke-{route}.json', dict(result, status='PASS', steps=1))
    else:
        torch.save(dict(proof=proof,models={k:m.state_dict() for k,m in models.items()}),
                   root/f'{route}-model.pt')
        write(root/f'{route}-complete.json', result)


@torch.no_grad()
def render(iq, noise, baseline, tracker, models, *, frozen):
    reference, active = baseline(iq)
    outputs = [reference.clone() for _ in MODES[2:]]
    if active.any():
        y, rotation, q, rms = coordinates(tracker, iq[active])
        basis = rotation*rms
        b = complex_iq(reference[active])/basis
        update = frozen(y,b,q)
        features, anchor = common_inputs(y,b,update,q,rms)
        ordinary, fine = (m(features,anchor) for m in models)
        half = match_common(anchor,.5*(anchor+fine))
        exact = packet_mean(fft(y-complex_iq(noise[active])/basis-b), rms)
        best = match_common(anchor,exact)
        for waveform, shared in zip(outputs[:-1], (anchor,ordinary,fine,half,best), strict=True):
            delta = update + (shared-anchor)[:, None]
            waveform[active] = reference[active]+real_iq(torch.fft.ifft(delta, norm='ortho')*basis)
        outputs[-1][active] = normalized(iq[active]-noise[active], iq[active])
    return torch.stack((iq, reference, *outputs)), int(active.sum())


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+key, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args, run=RUN, runner=__spec__.name)
    else:
        rt = read(args['runtime'])
        if action=='dispatch':
            dispatch(rt)
        else:
            activate(rt)
            route = args['route']
            if action=='score':
                binding = source_control(rt,route,'cuda:0')
                fixed = parent(Path(rt['run_root']),route,binding[2],'cuda:0')
                score(rt, route, recipe=RECIPE, panel=panel, modes=MODES,
                      render=partial(render,frozen=fixed), binding=binding)
            else:
                source_trace_implementation()
                fit(rt,route,smoke=action=='smoke')
