"""Fit a near-carrier refiner on two noise draws of each lawful source signal."""

import argparse
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .broadband_partitions_source import panel as swapped_panel
from .broadband_vector import target_vectors
from .broadband_vector_source import RECIPE as BROADBAND, RUN as PARENT, panel as broad_panel
from .complex_vector_source import dispatch, load_models, publish, record, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX, normalized
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .near_spectrum import NearSpectrum, matched_near, near_error, same_signal
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .signal_retention_source import build_arrays
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-near-spectrum-source-m2-r01'
RECIPE = SimpleNamespace(seed=2026112784, steps=1600, train_per_rx=700, held_per_rx=80,
                         arms=('direct', 'paired'), model=lambda _: NearSpectrum())
PAIR_WEIGHT = .12
BATCH = 8
MODES = ('direct', 'reference', 'frozen_broad', 'direct_near', 'paired_near',
         'paired_near_half', 'known_matched_near', 'known_noise')


def broadband(root, route, contract, device):
    parent = root.parent/PARENT
    if read(parent/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Require completed broadband weights')
    _, model = load_models(parent, route, contract, BROADBAND, device)
    return model


@torch.no_grad()
def training_pairs(raw, wide):
    observed, quality, signal, base, active, _, _, common = raw
    chosen = active.all(dim=1)
    y, b, q, truth, align = (a[chosen] for a in (observed, base, quality, signal, common))
    if len(y) < BATCH:
        raise ValueError('Need paired practical exposures under the same validated gate')
    rows = zip(*(v.flatten(0, 1).split(24) for v in (y, b, q)), strict=True)
    prior = torch.cat([wide(measured, restored, value) for measured, restored, value in rows])
    return y, b, q, truth, align, prior.unflatten(0, (len(y), 2))


def losses(model, arrays, indexes):
    y, b, q, signal, common, prior = (x[indexes] for x in arrays)
    prediction = model(*(v.flatten(0, 1) for v in (y, b, q, prior))).reshape_as(prior)
    return torch.stack((near_error(prediction, b, signal), same_signal(prediction, b, signal, common)))


def fit(rt, route, *, smoke=False, recipe=RECIPE, pair_weights=(0., PAIR_WEIGHT)):
    torch.manual_seed(recipe.seed)
    device = 'cuda:0'
    root = Path(rt['run_root'])
    _, _, contract, baseline, tracker = source_control(rt, route, device)
    wide = broadband(root, route, contract, device)
    count, held = (6, 4) if smoke else (recipe.train_per_rx, recipe.held_per_rx)
    fit_data, hold_data, archive_contract = build_arrays(rt, route, tracker, count, held,
                                                        baseline=baseline, with_frames=True)
    if contract != archive_contract:
        raise ValueError('Expected equal frozen waveform and fitting contract')
    samples, evaluation = (training_pairs(data, wide) for data in (fit_data, hold_data))
    del fit_data, hold_data
    arms = {k: recipe.model(k).to(device) for k in recipe.arms}
    first, second = arms.values()
    second.load_state_dict(first.state_dict())
    optimizer = torch.optim.AdamW([p for m in arms.values() for p in m.parameters()],
                                  lr=1.5e-4, weight_decay=1e-4)
    steps = 1 if smoke else recipe.steps
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=2e-5)
    generator = torch.Generator(device=device).manual_seed(recipe.seed)
    weights = torch.tensor(pair_weights, device=device)
    for step in range(1, steps+1):
        ix = torch.randint(len(samples[0]), (BATCH,), generator=generator, device=device)
        values = torch.stack([losses(model, samples, ix) for model in arms.values()])
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite near-carrier reconstruction')
        optimizer.zero_grad(set_to_none=True)
        (values[:, 0].sum()+(weights*values[:, 1]).sum()).backward()
        for model in arms.values():
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and (norm <= 0 or any(x.out.weight.grad.norm() <= 0
                                                for x in (model.spectral, model.temporal))):
                raise ValueError('Both near-carrier paths need legal source gradients')
        optimizer.step()
        lr.step()
        if step == 1 or step % 50 == 0:
            print('NEAR_SPECTRUM_FIT', route, step, *values.flatten().detach().tolist(), flush=True)
    ys, base, q, signal, common, prior = evaluation
    with torch.no_grad():
        held_loss = {}
        for k, model in arms.items():
            model.eval()
            indices = torch.arange(len(ys), device=device).split(BATCH)
            measured = torch.stack([len(i)*losses(model, evaluation, i) for i in indices]).sum(0)
            held_loss[k] = (measured/len(ys)).tolist()
        frozen_loss = [float(near_error(prior, base, signal)), float(same_signal(prior, base, signal, common))]
    proof = record(contract, route, recipe)
    report = {key: val for key, val in proof.items() if key!='contract'}
    report.update(fit_pairs=len(samples[0]), held_pairs=len(ys), held_loss=held_loss,
                  frozen_loss=frozen_loss, broadband_parent=PARENT, paired_weight=pair_weights[1],
                  pair_weights=list(pair_weights))
    if smoke:
        write(root/f'smoke-{route}.json', dict(report, status='PASS', steps=1,
                                               train_per_rx=count, held_per_rx=held))
    else:
        torch.save(dict(proof=proof, models={k:v.state_dict() for k,v in arms.items()}), root/f'{route}-model.pt')
        write(root/f'{route}-complete.json', report)


def panel(rx):
    prior = np.union1d(broad_panel(rx), swapped_panel(rx))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), prior),
                                             EVAL_PER_RX, replace=False)) for name in SOURCE])


@torch.no_grad()
def readouts(received, noise, baseline, tracker, models, *, wide):
    mature, active = baseline(received)
    rows = [mature.clone() for _ in MODES[2:]]
    if active.any():
        observed = received[active]
        frame, rotation, q, rms = coordinates(tracker, observed)
        basis = rotation*rms
        b = complex_iq(mature[active])/basis
        frozen = wide(frame, b, q)
        d, p = (m(frame, b, q, frozen) for m in models)
        half = matched_near(frozen, .5*(frozen+p))
        source_truth = frame-complex_iq(noise[active])/basis
        known = target_vectors(b, source_truth)['broad']
        reference = matched_near(frozen, known)
        for waveform, delta in zip(rows[:-1], (frozen, d, p, half, reference), strict=True):
            waveform[active] = mature[active] + real_iq(torch.fft.ifft(delta, norm='ortho')*basis)
        rows[-1][active] = normalized(observed-noise[active], observed)
    return torch.stack((received, mature, *rows)), int(active.sum())


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
    opts = vars(cli.parse_args())
    action = opts.pop('action')
    if action == 'publish':
        publish(**opts, run=RUN, runner=__spec__.name)
    else:
        rt = read(opts['runtime'])
        if action == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            route = opts['route']
            if action == 'score':
                context = source_control(rt, route, 'cuda:0')
                model = broadband(Path(rt['run_root']), route, context[2], 'cuda:0')
                score(rt, route, recipe=RECIPE, panel=panel, modes=MODES,
                      render=partial(readouts, wide=model), binding=context)
            else:
                source_trace_implementation()
                fit(rt, route, smoke=action=='smoke')
